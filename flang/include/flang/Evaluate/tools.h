//===-- include/flang/Evaluate/tools.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_TOOLS_H_
#define FORTRAN_EVALUATE_TOOLS_H_

#include "traverse.h"
#include "flang/Common/idioms.h"
#include "flang/Common/template.h"
#include "flang/Common/unwrap.h"
#include "flang/Evaluate/constant.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/shape.h"
#include "flang/Evaluate/type.h"
#include "flang/Parser/message.h"
#include "flang/Semantics/attr.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/symbol.h"
#include <array>
#include <optional>
#include <set>
#include <type_traits>
#include <utility>

namespace Fortran::evaluate {

// Some expression predicates and extractors.

// Predicate: true when an expression is a variable reference, not an
// operation.  Be advised: a call to a function that returns an object
// pointer is a "variable" in Fortran (it can be the left-hand side of
// an assignment).
struct IsVariableHelper
    : public AnyTraverse<IsVariableHelper, std::optional<bool>> {
  using Result = std::optional<bool>; // effectively tri-state
  using Base = AnyTraverse<IsVariableHelper, Result>;
  IsVariableHelper() : Base{*this} {}
  using Base::operator();
  Result operator()(const StaticDataObject &) const { return false; }
  Result operator()(const Symbol &) const;
  Result operator()(const Component &) const;
  Result operator()(const ArrayRef &) const;
  Result operator()(const Substring &) const;
  Result operator()(const CoarrayRef &) const { return true; }
  Result operator()(const ComplexPart &) const { return true; }
  Result operator()(const ProcedureDesignator &) const;
  template <typename T> Result operator()(const Expr<T> &x) const {
    if constexpr (common::HasMember<T, AllIntrinsicTypes> ||
        std::is_same_v<T, SomeDerived>) {
      // Expression with a specific type
      if (std::holds_alternative<Designator<T>>(x.u) ||
          std::holds_alternative<FunctionRef<T>>(x.u)) {
        if (auto known{(*this)(x.u)}) {
          return known;
        }
      }
      return false;
    } else if constexpr (std::is_same_v<T, SomeType>) {
      if (std::holds_alternative<ProcedureDesignator>(x.u) ||
          std::holds_alternative<ProcedureRef>(x.u)) {
        return false; // procedure pointer
      } else {
        return (*this)(x.u);
      }
    } else {
      return (*this)(x.u);
    }
  }
};

template <typename A> bool IsVariable(const A &x) {
  if (auto known{IsVariableHelper{}(x)}) {
    return *known;
  } else {
    return false;
  }
}

// Predicate: true when an expression is assumed-rank
bool IsAssumedRank(const Symbol &);
bool IsAssumedRank(const ActualArgument &);
template <typename A> bool IsAssumedRank(const A &) { return false; }
template <typename A> bool IsAssumedRank(const Designator<A> &designator) {
  if (const auto *symbol{std::get_if<SymbolRef>(&designator.u)}) {
    return IsAssumedRank(symbol->get());
  } else {
    return false;
  }
}
template <typename T> bool IsAssumedRank(const Expr<T> &expr) {
  return common::visit([](const auto &x) { return IsAssumedRank(x); }, expr.u);
}
template <typename A> bool IsAssumedRank(const std::optional<A> &x) {
  return x && IsAssumedRank(*x);
}
template <typename A> bool IsAssumedRank(const A *x) {
  return x && IsAssumedRank(*x);
}

// Finds the corank of an entity, possibly packaged in various ways.
// Unlike rank, only data references have corank > 0.
int GetCorank(const ActualArgument &);
static inline int GetCorank(const Symbol &symbol) { return symbol.Corank(); }
template <typename A> int GetCorank(const A &) { return 0; }
template <typename T> int GetCorank(const Designator<T> &designator) {
  return designator.Corank();
}
template <typename T> int GetCorank(const Expr<T> &expr) {
  return common::visit([](const auto &x) { return GetCorank(x); }, expr.u);
}
template <typename A> int GetCorank(const std::optional<A> &x) {
  return x ? GetCorank(*x) : 0;
}
template <typename A> int GetCorank(const A *x) {
  return x ? GetCorank(*x) : 0;
}

// Predicate: true when an expression is a coarray (corank > 0)
template <typename A> bool IsCoarray(const A &x) { return GetCorank(x) > 0; }

// Generalizing packagers: these take operations and expressions of more
// specific types and wrap them in Expr<> containers of more abstract types.

template <typename A> common::IfNoLvalue<Expr<ResultType<A>>, A> AsExpr(A &&x) {
  return Expr<ResultType<A>>{std::move(x)};
}

template <typename T> Expr<T> AsExpr(Expr<T> &&x) {
  static_assert(IsSpecificIntrinsicType<T>);
  return std::move(x);
}

template <TypeCategory CATEGORY>
Expr<SomeKind<CATEGORY>> AsCategoryExpr(Expr<SomeKind<CATEGORY>> &&x) {
  return std::move(x);
}

template <typename A>
common::IfNoLvalue<Expr<SomeType>, A> AsGenericExpr(A &&x) {
  if constexpr (common::HasMember<A, TypelessExpression>) {
    return Expr<SomeType>{std::move(x)};
  } else {
    return Expr<SomeType>{AsCategoryExpr(std::move(x))};
  }
}

inline Expr<SomeType> AsGenericExpr(Expr<SomeType> &&x) { return std::move(x); }

// These overloads wrap DataRefs and simple whole variables up into
// generic expressions if they have a known type.
std::optional<Expr<SomeType>> AsGenericExpr(DataRef &&);
std::optional<Expr<SomeType>> AsGenericExpr(const Symbol &);

// Propagate std::optional from input to output.
template <typename A>
std::optional<Expr<SomeType>> AsGenericExpr(std::optional<A> &&x) {
  if (x) {
    return AsGenericExpr(std::move(*x));
  } else {
    return std::nullopt;
  }
}

template <typename A>
common::IfNoLvalue<Expr<SomeKind<ResultType<A>::category>>, A> AsCategoryExpr(
    A &&x) {
  return Expr<SomeKind<ResultType<A>::category>>{AsExpr(std::move(x))};
}

Expr<SomeType> Parenthesize(Expr<SomeType> &&);

template <typename A> constexpr bool IsNumericCategoryExpr() {
  if constexpr (common::HasMember<A, TypelessExpression>) {
    return false;
  } else {
    return common::HasMember<ResultType<A>, NumericCategoryTypes>;
  }
}

// Specializing extractor.  If an Expr wraps some type of object, perhaps
// in several layers, return a pointer to it; otherwise null.  Also works
// with expressions contained in ActualArgument.
template <typename A, typename B>
auto UnwrapExpr(B &x) -> common::Constify<A, B> * {
  using Ty = std::decay_t<B>;
  if constexpr (std::is_same_v<A, Ty>) {
    return &x;
  } else if constexpr (std::is_same_v<Ty, ActualArgument>) {
    if (auto *expr{x.UnwrapExpr()}) {
      return UnwrapExpr<A>(*expr);
    }
  } else if constexpr (std::is_same_v<Ty, Expr<SomeType>>) {
    return common::visit([](auto &x) { return UnwrapExpr<A>(x); }, x.u);
  } else if constexpr (!common::HasMember<A, TypelessExpression>) {
    if constexpr (std::is_same_v<Ty, Expr<ResultType<A>>> ||
        std::is_same_v<Ty, Expr<SomeKind<ResultType<A>::category>>>) {
      return common::visit([](auto &x) { return UnwrapExpr<A>(x); }, x.u);
    }
  }
  return nullptr;
}

template <typename A, typename B>
const A *UnwrapExpr(const std::optional<B> &x) {
  if (x) {
    return UnwrapExpr<A>(*x);
  } else {
    return nullptr;
  }
}

template <typename A, typename B> A *UnwrapExpr(std::optional<B> &x) {
  if (x) {
    return UnwrapExpr<A>(*x);
  } else {
    return nullptr;
  }
}

template <typename A, typename B> const A *UnwrapExpr(const B *x) {
  if (x) {
    return UnwrapExpr<A>(*x);
  } else {
    return nullptr;
  }
}

template <typename A, typename B> A *UnwrapExpr(B *x) {
  if (x) {
    return UnwrapExpr<A>(*x);
  } else {
    return nullptr;
  }
}

// A variant of UnwrapExpr above that also skips through (parentheses)
// and conversions of kinds within a category.  Useful for extracting LEN
// type parameter inquiries, at least.
template <typename A, typename B>
auto UnwrapConvertedExpr(B &x) -> common::Constify<A, B> * {
  using Ty = std::decay_t<B>;
  if constexpr (std::is_same_v<A, Ty>) {
    return &x;
  } else if constexpr (std::is_same_v<Ty, ActualArgument>) {
    if (auto *expr{x.UnwrapExpr()}) {
      return UnwrapConvertedExpr<A>(*expr);
    }
  } else if constexpr (std::is_same_v<Ty, Expr<SomeType>>) {
    return common::visit(
        [](auto &x) { return UnwrapConvertedExpr<A>(x); }, x.u);
  } else {
    using DesiredResult = ResultType<A>;
    if constexpr (std::is_same_v<Ty, Expr<DesiredResult>> ||
        std::is_same_v<Ty, Expr<SomeKind<DesiredResult::category>>>) {
      return common::visit(
          [](auto &x) { return UnwrapConvertedExpr<A>(x); }, x.u);
    } else {
      using ThisResult = ResultType<B>;
      if constexpr (std::is_same_v<Ty, Expr<ThisResult>>) {
        return common::visit(
            [](auto &x) { return UnwrapConvertedExpr<A>(x); }, x.u);
      } else if constexpr (std::is_same_v<Ty, Parentheses<ThisResult>> ||
          std::is_same_v<Ty, Convert<ThisResult, DesiredResult::category>>) {
        return common::visit(
            [](auto &x) { return UnwrapConvertedExpr<A>(x); }, x.left().u);
      }
    }
  }
  return nullptr;
}

// UnwrapProcedureRef() returns a pointer to a ProcedureRef when the whole
// expression is a reference to a procedure.
template <typename A> inline const ProcedureRef *UnwrapProcedureRef(const A &) {
  return nullptr;
}

inline const ProcedureRef *UnwrapProcedureRef(const ProcedureRef &proc) {
  // Reference to subroutine or to a function that returns
  // an object pointer or procedure pointer
  return &proc;
}

template <typename T>
inline const ProcedureRef *UnwrapProcedureRef(const FunctionRef<T> &func) {
  return &func; // reference to a function returning a non-pointer
}

template <typename T>
inline const ProcedureRef *UnwrapProcedureRef(const Expr<T> &expr) {
  return common::visit(
      [](const auto &x) { return UnwrapProcedureRef(x); }, expr.u);
}

// When an expression is a "bare" LEN= derived type parameter inquiry,
// possibly wrapped in integer kind conversions &/or parentheses, return
// a pointer to the Symbol with TypeParamDetails.
template <typename A> const Symbol *ExtractBareLenParameter(const A &expr) {
  if (const auto *typeParam{
          UnwrapConvertedExpr<evaluate::TypeParamInquiry>(expr)}) {
    if (!typeParam->base()) {
      const Symbol &symbol{typeParam->parameter()};
      if (const auto *tpd{symbol.detailsIf<semantics::TypeParamDetails>()}) {
        if (tpd->attr() == common::TypeParamAttr::Len) {
          return &symbol;
        }
      }
    }
  }
  return nullptr;
}

// If an expression simply wraps a DataRef, extract and return it.
// The Boolean arguments control the handling of Substring and ComplexPart
// references: when true (not default), it extracts the base DataRef
// of a substring or complex part.
template <typename A>
common::IfNoLvalue<std::optional<DataRef>, A> ExtractDataRef(
    const A &, bool intoSubstring, bool intoComplexPart) {
  return std::nullopt; // default base case
}
template <typename T>
std::optional<DataRef> ExtractDataRef(const Designator<T> &d,
    bool intoSubstring = false, bool intoComplexPart = false) {
  return common::visit(
      [=](const auto &x) -> std::optional<DataRef> {
        if constexpr (common::HasMember<decltype(x), decltype(DataRef::u)>) {
          return DataRef{x};
        }
        if constexpr (std::is_same_v<std::decay_t<decltype(x)>, Substring>) {
          if (intoSubstring) {
            return ExtractSubstringBase(x);
          }
        }
        if constexpr (std::is_same_v<std::decay_t<decltype(x)>, ComplexPart>) {
          if (intoComplexPart) {
            return x.complex();
          }
        }
        return std::nullopt; // w/o "else" to dodge bogus g++ 8.1 warning
      },
      d.u);
}
template <typename T>
std::optional<DataRef> ExtractDataRef(const Expr<T> &expr,
    bool intoSubstring = false, bool intoComplexPart = false) {
  return common::visit(
      [=](const auto &x) {
        return ExtractDataRef(x, intoSubstring, intoComplexPart);
      },
      expr.u);
}
template <typename A>
std::optional<DataRef> ExtractDataRef(const std::optional<A> &x,
    bool intoSubstring = false, bool intoComplexPart = false) {
  if (x) {
    return ExtractDataRef(*x, intoSubstring, intoComplexPart);
  } else {
    return std::nullopt;
  }
}
template <typename A>
std::optional<DataRef> ExtractDataRef(
    A *p, bool intoSubstring = false, bool intoComplexPart = false) {
  if (p) {
    return ExtractDataRef(std::as_const(*p), intoSubstring, intoComplexPart);
  } else {
    return std::nullopt;
  }
}
std::optional<DataRef> ExtractDataRef(const ActualArgument &,
    bool intoSubstring = false, bool intoComplexPart = false);

std::optional<DataRef> ExtractSubstringBase(const Substring &);

// Predicate: is an expression is an array element reference?
template <typename T>
bool IsArrayElement(const Expr<T> &expr, bool intoSubstring = true,
    bool skipComponents = false) {
  if (auto dataRef{ExtractDataRef(expr, intoSubstring)}) {
    const DataRef *ref{&*dataRef};
    if (skipComponents) {
      while (const Component * component{std::get_if<Component>(&ref->u)}) {
        ref = &component->base();
      }
    }
    if (const auto *coarrayRef{std::get_if<CoarrayRef>(&ref->u)}) {
      return !coarrayRef->subscript().empty();
    } else {
      return std::holds_alternative<ArrayRef>(ref->u);
    }
  } else {
    return false;
  }
}

template <typename A>
std::optional<NamedEntity> ExtractNamedEntity(const A &x) {
  if (auto dataRef{ExtractDataRef(x)}) {
    return common::visit(
        common::visitors{
            [](SymbolRef &&symbol) -> std::optional<NamedEntity> {
              return NamedEntity{symbol};
            },
            [](Component &&component) -> std::optional<NamedEntity> {
              return NamedEntity{std::move(component)};
            },
            [](CoarrayRef &&co) -> std::optional<NamedEntity> {
              return co.GetBase();
            },
            [](auto &&) { return std::optional<NamedEntity>{}; },
        },
        std::move(dataRef->u));
  } else {
    return std::nullopt;
  }
}

struct ExtractCoindexedObjectHelper {
  template <typename A> std::optional<CoarrayRef> operator()(const A &) const {
    return std::nullopt;
  }
  std::optional<CoarrayRef> operator()(const CoarrayRef &x) const { return x; }
  template <typename A>
  std::optional<CoarrayRef> operator()(const Expr<A> &expr) const {
    return common::visit(*this, expr.u);
  }
  std::optional<CoarrayRef> operator()(const DataRef &dataRef) const {
    return common::visit(*this, dataRef.u);
  }
  std::optional<CoarrayRef> operator()(const NamedEntity &named) const {
    if (const Component * component{named.UnwrapComponent()}) {
      return (*this)(*component);
    } else {
      return std::nullopt;
    }
  }
  std::optional<CoarrayRef> operator()(const ProcedureDesignator &des) const {
    if (const auto *component{
            std::get_if<common::CopyableIndirection<Component>>(&des.u)}) {
      return (*this)(component->value());
    } else {
      return std::nullopt;
    }
  }
  std::optional<CoarrayRef> operator()(const Component &component) const {
    return (*this)(component.base());
  }
  std::optional<CoarrayRef> operator()(const ArrayRef &arrayRef) const {
    return (*this)(arrayRef.base());
  }
};

template <typename A> std::optional<CoarrayRef> ExtractCoarrayRef(const A &x) {
  if (auto dataRef{ExtractDataRef(x, true)}) {
    return ExtractCoindexedObjectHelper{}(*dataRef);
  } else {
    return ExtractCoindexedObjectHelper{}(x);
  }
}

struct ExtractSubstringHelper {
  template <typename T> static std::optional<Substring> visit(T &&) {
    return std::nullopt;
  }

  static std::optional<Substring> visit(const Substring &e) { return e; }

  template <typename T>
  static std::optional<Substring> visit(const Designator<T> &e) {
    return common::visit([](auto &&s) { return visit(s); }, e.u);
  }

  template <typename T>
  static std::optional<Substring> visit(const Expr<T> &e) {
    return common::visit([](auto &&s) { return visit(s); }, e.u);
  }
};

template <typename A> std::optional<Substring> ExtractSubstring(const A &x) {
  return ExtractSubstringHelper::visit(x);
}

// If an expression is simply a whole symbol data designator,
// extract and return that symbol, else null.
template <typename A> const Symbol *UnwrapWholeSymbolDataRef(const A &x) {
  if (auto dataRef{ExtractDataRef(x)}) {
    if (const SymbolRef * p{std::get_if<SymbolRef>(&dataRef->u)}) {
      return &p->get();
    }
  }
  return nullptr;
}

// If an expression is a whole symbol or a whole component desginator,
// extract and return that symbol, else null.
template <typename A>
const Symbol *UnwrapWholeSymbolOrComponentDataRef(const A &x) {
  if (auto dataRef{ExtractDataRef(x)}) {
    if (const SymbolRef * p{std::get_if<SymbolRef>(&dataRef->u)}) {
      return &p->get();
    } else if (const Component * c{std::get_if<Component>(&dataRef->u)}) {
      if (c->base().Rank() == 0) {
        return &c->GetLastSymbol();
      }
    }
  }
  return nullptr;
}

// If an expression is a whole symbol or a whole component designator,
// potentially followed by an image selector, extract and return that symbol,
// else null.
template <typename A>
const Symbol *UnwrapWholeSymbolOrComponentOrCoarrayRef(const A &x) {
  if (auto dataRef{ExtractDataRef(x)}) {
    if (const SymbolRef * p{std::get_if<SymbolRef>(&dataRef->u)}) {
      return &p->get();
    } else if (const Component * c{std::get_if<Component>(&dataRef->u)}) {
      if (c->base().Rank() == 0) {
        return &c->GetLastSymbol();
      }
    } else if (const CoarrayRef * c{std::get_if<CoarrayRef>(&dataRef->u)}) {
      if (c->subscript().empty()) {
        return &c->GetLastSymbol();
      }
    }
  }
  return nullptr;
}

// GetFirstSymbol(A%B%C[I]%D) -> A
template <typename A> const Symbol *GetFirstSymbol(const A &x) {
  if (auto dataRef{ExtractDataRef(x, true)}) {
    return &dataRef->GetFirstSymbol();
  } else {
    return nullptr;
  }
}

// GetLastPointerSymbol(A%PTR1%B%PTR2%C) -> PTR2
const Symbol *GetLastPointerSymbol(const evaluate::DataRef &);

// Creation of conversion expressions can be done to either a known
// specific intrinsic type with ConvertToType<T>(x) or by converting
// one arbitrary expression to the type of another with ConvertTo(to, from).

template <typename TO, TypeCategory FROMCAT>
Expr<TO> ConvertToType(Expr<SomeKind<FROMCAT>> &&x) {
  static_assert(IsSpecificIntrinsicType<TO>);
  if constexpr (FROMCAT == TO::category) {
    if (auto *already{std::get_if<Expr<TO>>(&x.u)}) {
      return std::move(*already);
    } else {
      return Expr<TO>{Convert<TO, FROMCAT>{std::move(x)}};
    }
  } else if constexpr (TO::category == TypeCategory::Complex) {
    using Part = typename TO::Part;
    Scalar<Part> zero;
    return Expr<TO>{ComplexConstructor<TO::kind>{
        ConvertToType<Part>(std::move(x)), Expr<Part>{Constant<Part>{zero}}}};
  } else if constexpr (FROMCAT == TypeCategory::Complex) {
    // Extract and convert the real component of a complex value
    return common::visit(
        [&](auto &&z) {
          using ZType = ResultType<decltype(z)>;
          using Part = typename ZType::Part;
          return ConvertToType<TO, TypeCategory::Real>(Expr<SomeReal>{
              Expr<Part>{ComplexComponent<Part::kind>{false, std::move(z)}}});
        },
        std::move(x.u));
  } else {
    return Expr<TO>{Convert<TO, FROMCAT>{std::move(x)}};
  }
}

template <typename TO, TypeCategory FROMCAT, int FROMKIND>
Expr<TO> ConvertToType(Expr<Type<FROMCAT, FROMKIND>> &&x) {
  return ConvertToType<TO, FROMCAT>(Expr<SomeKind<FROMCAT>>{std::move(x)});
}

template <typename TO> Expr<TO> ConvertToType(BOZLiteralConstant &&x) {
  static_assert(IsSpecificIntrinsicType<TO>);
  if constexpr (TO::category == TypeCategory::Integer ||
      TO::category == TypeCategory::Unsigned) {
    return Expr<TO>{
        Constant<TO>{Scalar<TO>::ConvertUnsigned(std::move(x)).value}};
  } else {
    static_assert(TO::category == TypeCategory::Real);
    using Word = typename Scalar<TO>::Word;
    return Expr<TO>{
        Constant<TO>{Scalar<TO>{Word::ConvertUnsigned(std::move(x)).value}}};
  }
}

template <typename T> bool IsBOZLiteral(const Expr<T> &expr) {
  return std::holds_alternative<BOZLiteralConstant>(expr.u);
}

// Conversions to dynamic types
std::optional<Expr<SomeType>> ConvertToType(
    const DynamicType &, Expr<SomeType> &&);
std::optional<Expr<SomeType>> ConvertToType(
    const DynamicType &, std::optional<Expr<SomeType>> &&);
std::optional<Expr<SomeType>> ConvertToType(const Symbol &, Expr<SomeType> &&);
std::optional<Expr<SomeType>> ConvertToType(
    const Symbol &, std::optional<Expr<SomeType>> &&);

// Conversions to the type of another expression
template <TypeCategory TC, int TK, typename FROM>
common::IfNoLvalue<Expr<Type<TC, TK>>, FROM> ConvertTo(
    const Expr<Type<TC, TK>> &, FROM &&x) {
  return ConvertToType<Type<TC, TK>>(std::move(x));
}

template <TypeCategory TC, typename FROM>
common::IfNoLvalue<Expr<SomeKind<TC>>, FROM> ConvertTo(
    const Expr<SomeKind<TC>> &to, FROM &&from) {
  return common::visit(
      [&](const auto &toKindExpr) {
        using KindExpr = std::decay_t<decltype(toKindExpr)>;
        return AsCategoryExpr(
            ConvertToType<ResultType<KindExpr>>(std::move(from)));
      },
      to.u);
}

template <typename FROM>
common::IfNoLvalue<Expr<SomeType>, FROM> ConvertTo(
    const Expr<SomeType> &to, FROM &&from) {
  return common::visit(
      [&](const auto &toCatExpr) {
        return AsGenericExpr(ConvertTo(toCatExpr, std::move(from)));
      },
      to.u);
}

// Convert an expression of some known category to a dynamically chosen
// kind of some category (usually but not necessarily distinct).
template <TypeCategory TOCAT, typename VALUE> struct ConvertToKindHelper {
  using Result = std::optional<Expr<SomeKind<TOCAT>>>;
  using Types = CategoryTypes<TOCAT>;
  ConvertToKindHelper(int k, VALUE &&x) : kind{k}, value{std::move(x)} {}
  template <typename T> Result Test() {
    if (kind == T::kind) {
      return std::make_optional(
          AsCategoryExpr(ConvertToType<T>(std::move(value))));
    }
    return std::nullopt;
  }
  int kind;
  VALUE value;
};

template <TypeCategory TOCAT, typename VALUE>
common::IfNoLvalue<Expr<SomeKind<TOCAT>>, VALUE> ConvertToKind(
    int kind, VALUE &&x) {
  auto result{common::SearchTypes(
      ConvertToKindHelper<TOCAT, VALUE>{kind, std::move(x)})};
  CHECK(result.has_value());
  return *result;
}

// Given a type category CAT, SameKindExprs<CAT, N> is a variant that
// holds an arrays of expressions of the same supported kind in that
// category.
template <typename A, int N = 2> using SameExprs = std::array<Expr<A>, N>;
template <int N = 2> struct SameKindExprsHelper {
  template <typename A> using SameExprs = std::array<Expr<A>, N>;
};
template <TypeCategory CAT, int N = 2>
using SameKindExprs =
    common::MapTemplate<SameKindExprsHelper<N>::template SameExprs,
        CategoryTypes<CAT>>;

// Given references to two expressions of arbitrary kind in the same type
// category, convert one to the kind of the other when it has the smaller kind,
// then return them in a type-safe package.
template <TypeCategory CAT>
SameKindExprs<CAT, 2> AsSameKindExprs(
    Expr<SomeKind<CAT>> &&x, Expr<SomeKind<CAT>> &&y) {
  return common::visit(
      [&](auto &&kx, auto &&ky) -> SameKindExprs<CAT, 2> {
        using XTy = ResultType<decltype(kx)>;
        using YTy = ResultType<decltype(ky)>;
        if constexpr (std::is_same_v<XTy, YTy>) {
          return {SameExprs<XTy>{std::move(kx), std::move(ky)}};
        } else if constexpr (XTy::kind < YTy::kind) {
          return {SameExprs<YTy>{ConvertTo(ky, std::move(kx)), std::move(ky)}};
        } else {
          return {SameExprs<XTy>{std::move(kx), ConvertTo(kx, std::move(ky))}};
        }
#if !__clang__ && 100 * __GNUC__ + __GNUC_MINOR__ == 801
        // Silence a bogus warning about a missing return with G++ 8.1.0.
        // Doesn't execute, but must be correctly typed.
        CHECK(!"can't happen");
        return {SameExprs<XTy>{std::move(kx), std::move(kx)}};
#endif
      },
      std::move(x.u), std::move(y.u));
}

// Ensure that both operands of an intrinsic REAL operation (or CMPLX()
// constructor) are INTEGER or REAL, then convert them as necessary to the
// same kind of REAL.
using ConvertRealOperandsResult =
    std::optional<SameKindExprs<TypeCategory::Real, 2>>;
ConvertRealOperandsResult ConvertRealOperands(parser::ContextualMessages &,
    Expr<SomeType> &&, Expr<SomeType> &&, int defaultRealKind);

// Per F'2018 R718, if both components are INTEGER, they are both converted
// to default REAL and the result is default COMPLEX.  Otherwise, the
// kind of the result is the kind of most precise REAL component, and the other
// component is converted if necessary to its type.
std::optional<Expr<SomeComplex>> ConstructComplex(parser::ContextualMessages &,
    Expr<SomeType> &&, Expr<SomeType> &&, int defaultRealKind);
std::optional<Expr<SomeComplex>> ConstructComplex(parser::ContextualMessages &,
    std::optional<Expr<SomeType>> &&, std::optional<Expr<SomeType>> &&,
    int defaultRealKind);

template <typename A> Expr<TypeOf<A>> ScalarConstantToExpr(const A &x) {
  using Ty = TypeOf<A>;
  static_assert(
      std::is_same_v<Scalar<Ty>, std::decay_t<A>>, "TypeOf<> is broken");
  return Expr<TypeOf<A>>{Constant<Ty>{x}};
}

// Combine two expressions of the same specific numeric type with an operation
// to produce a new expression.
template <template <typename> class OPR, typename SPECIFIC>
Expr<SPECIFIC> Combine(Expr<SPECIFIC> &&x, Expr<SPECIFIC> &&y) {
  static_assert(IsSpecificIntrinsicType<SPECIFIC>);
  return AsExpr(OPR<SPECIFIC>{std::move(x), std::move(y)});
}

// Given two expressions of arbitrary kind in the same intrinsic type
// category, convert one of them if necessary to the larger kind of the
// other, then combine the resulting homogenized operands with a given
// operation, returning a new expression in the same type category.
template <template <typename> class OPR, TypeCategory CAT>
Expr<SomeKind<CAT>> PromoteAndCombine(
    Expr<SomeKind<CAT>> &&x, Expr<SomeKind<CAT>> &&y) {
  return common::visit(
      [](auto &&xy) {
        using Ty = ResultType<decltype(xy[0])>;
        return AsCategoryExpr(
            Combine<OPR, Ty>(std::move(xy[0]), std::move(xy[1])));
      },
      AsSameKindExprs(std::move(x), std::move(y)));
}

// Given two expressions of arbitrary type, try to combine them with a
// binary numeric operation (e.g., Add), possibly with data type conversion of
// one of the operands to the type of the other.  Handles special cases with
// typeless literal operands and with REAL/COMPLEX exponentiation to INTEGER
// powers.
template <template <typename> class OPR, bool CAN_BE_UNSIGNED = true>
std::optional<Expr<SomeType>> NumericOperation(parser::ContextualMessages &,
    Expr<SomeType> &&, Expr<SomeType> &&, int defaultRealKind);

extern template std::optional<Expr<SomeType>> NumericOperation<Power, false>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&,
    int defaultRealKind);
extern template std::optional<Expr<SomeType>> NumericOperation<Multiply>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&,
    int defaultRealKind);
extern template std::optional<Expr<SomeType>> NumericOperation<Divide>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&,
    int defaultRealKind);
extern template std::optional<Expr<SomeType>> NumericOperation<Add>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&,
    int defaultRealKind);
extern template std::optional<Expr<SomeType>> NumericOperation<Subtract>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&,
    int defaultRealKind);

std::optional<Expr<SomeType>> Negation(
    parser::ContextualMessages &, Expr<SomeType> &&);

// Given two expressions of arbitrary type, try to combine them with a
// relational operator (e.g., .LT.), possibly with data type conversion.
std::optional<Expr<LogicalResult>> Relate(parser::ContextualMessages &,
    RelationalOperator, Expr<SomeType> &&, Expr<SomeType> &&);

// Create a relational operation between two identically-typed operands
// and wrap it up in an Expr<LogicalResult>.
template <typename T>
Expr<LogicalResult> PackageRelation(
    RelationalOperator opr, Expr<T> &&x, Expr<T> &&y) {
  static_assert(IsSpecificIntrinsicType<T>);
  return Expr<LogicalResult>{
      Relational<SomeType>{Relational<T>{opr, std::move(x), std::move(y)}}};
}

template <int K>
Expr<Type<TypeCategory::Logical, K>> LogicalNegation(
    Expr<Type<TypeCategory::Logical, K>> &&x) {
  return AsExpr(Not<K>{std::move(x)});
}

Expr<SomeLogical> LogicalNegation(Expr<SomeLogical> &&);

template <int K>
Expr<Type<TypeCategory::Logical, K>> BinaryLogicalOperation(LogicalOperator opr,
    Expr<Type<TypeCategory::Logical, K>> &&x,
    Expr<Type<TypeCategory::Logical, K>> &&y) {
  return AsExpr(LogicalOperation<K>{opr, std::move(x), std::move(y)});
}

Expr<SomeLogical> BinaryLogicalOperation(
    LogicalOperator, Expr<SomeLogical> &&, Expr<SomeLogical> &&);

// Convenience functions and operator overloadings for expression construction.
// These interfaces are defined only for those situations that can never
// emit any message.  Use the more general templates (above) in other
// situations.

template <TypeCategory C, int K>
Expr<Type<C, K>> operator-(Expr<Type<C, K>> &&x) {
  return AsExpr(Negate<Type<C, K>>{std::move(x)});
}

template <TypeCategory C, int K>
Expr<Type<C, K>> operator+(Expr<Type<C, K>> &&x, Expr<Type<C, K>> &&y) {
  return AsExpr(Combine<Add, Type<C, K>>(std::move(x), std::move(y)));
}

template <TypeCategory C, int K>
Expr<Type<C, K>> operator-(Expr<Type<C, K>> &&x, Expr<Type<C, K>> &&y) {
  return AsExpr(Combine<Subtract, Type<C, K>>(std::move(x), std::move(y)));
}

template <TypeCategory C, int K>
Expr<Type<C, K>> operator*(Expr<Type<C, K>> &&x, Expr<Type<C, K>> &&y) {
  return AsExpr(Combine<Multiply, Type<C, K>>(std::move(x), std::move(y)));
}

template <TypeCategory C, int K>
Expr<Type<C, K>> operator/(Expr<Type<C, K>> &&x, Expr<Type<C, K>> &&y) {
  return AsExpr(Combine<Divide, Type<C, K>>(std::move(x), std::move(y)));
}

template <TypeCategory C> Expr<SomeKind<C>> operator-(Expr<SomeKind<C>> &&x) {
  return common::visit(
      [](auto &xk) { return Expr<SomeKind<C>>{-std::move(xk)}; }, x.u);
}

template <TypeCategory CAT>
Expr<SomeKind<CAT>> operator+(
    Expr<SomeKind<CAT>> &&x, Expr<SomeKind<CAT>> &&y) {
  return PromoteAndCombine<Add, CAT>(std::move(x), std::move(y));
}

template <TypeCategory CAT>
Expr<SomeKind<CAT>> operator-(
    Expr<SomeKind<CAT>> &&x, Expr<SomeKind<CAT>> &&y) {
  return PromoteAndCombine<Subtract, CAT>(std::move(x), std::move(y));
}

template <TypeCategory CAT>
Expr<SomeKind<CAT>> operator*(
    Expr<SomeKind<CAT>> &&x, Expr<SomeKind<CAT>> &&y) {
  return PromoteAndCombine<Multiply, CAT>(std::move(x), std::move(y));
}

template <TypeCategory CAT>
Expr<SomeKind<CAT>> operator/(
    Expr<SomeKind<CAT>> &&x, Expr<SomeKind<CAT>> &&y) {
  return PromoteAndCombine<Divide, CAT>(std::move(x), std::move(y));
}

// A utility for use with common::SearchTypes to create generic expressions
// when an intrinsic type category for (say) a variable is known
// but the kind parameter value is not.
template <TypeCategory CAT, template <typename> class TEMPLATE, typename VALUE>
struct TypeKindVisitor {
  using Result = std::optional<Expr<SomeType>>;
  using Types = CategoryTypes<CAT>;

  TypeKindVisitor(int k, VALUE &&x) : kind{k}, value{std::move(x)} {}
  TypeKindVisitor(int k, const VALUE &x) : kind{k}, value{x} {}

  template <typename T> Result Test() {
    if (kind == T::kind) {
      return AsGenericExpr(TEMPLATE<T>{std::move(value)});
    }
    return std::nullopt;
  }

  int kind;
  VALUE value;
};

// TypedWrapper() wraps a object in an explicitly typed representation
// (e.g., Designator<> or FunctionRef<>) that has been instantiated on
// a dynamically chosen Fortran type.
template <TypeCategory CATEGORY, template <typename> typename WRAPPER,
    typename WRAPPED>
common::IfNoLvalue<std::optional<Expr<SomeType>>, WRAPPED> WrapperHelper(
    int kind, WRAPPED &&x) {
  return common::SearchTypes(
      TypeKindVisitor<CATEGORY, WRAPPER, WRAPPED>{kind, std::move(x)});
}

template <template <typename> typename WRAPPER, typename WRAPPED>
common::IfNoLvalue<std::optional<Expr<SomeType>>, WRAPPED> TypedWrapper(
    const DynamicType &dyType, WRAPPED &&x) {
  switch (dyType.category()) {
    SWITCH_COVERS_ALL_CASES
  case TypeCategory::Integer:
    return WrapperHelper<TypeCategory::Integer, WRAPPER, WRAPPED>(
        dyType.kind(), std::move(x));
  case TypeCategory::Unsigned:
    return WrapperHelper<TypeCategory::Unsigned, WRAPPER, WRAPPED>(
        dyType.kind(), std::move(x));
  case TypeCategory::Real:
    return WrapperHelper<TypeCategory::Real, WRAPPER, WRAPPED>(
        dyType.kind(), std::move(x));
  case TypeCategory::Complex:
    return WrapperHelper<TypeCategory::Complex, WRAPPER, WRAPPED>(
        dyType.kind(), std::move(x));
  case TypeCategory::Character:
    return WrapperHelper<TypeCategory::Character, WRAPPER, WRAPPED>(
        dyType.kind(), std::move(x));
  case TypeCategory::Logical:
    return WrapperHelper<TypeCategory::Logical, WRAPPER, WRAPPED>(
        dyType.kind(), std::move(x));
  case TypeCategory::Derived:
    return AsGenericExpr(Expr<SomeDerived>{WRAPPER<SomeDerived>{std::move(x)}});
  }
}

// GetLastSymbol() returns the rightmost symbol in an object or procedure
// designator (which has perhaps been wrapped in an Expr<>), or a null pointer
// when none is found.  It will return an ASSOCIATE construct entity's symbol
// rather than descending into its expression.
struct GetLastSymbolHelper
    : public AnyTraverse<GetLastSymbolHelper, std::optional<const Symbol *>> {
  using Result = std::optional<const Symbol *>;
  using Base = AnyTraverse<GetLastSymbolHelper, Result>;
  GetLastSymbolHelper() : Base{*this} {}
  using Base::operator();
  Result operator()(const Symbol &x) const { return &x; }
  Result operator()(const Component &x) const { return &x.GetLastSymbol(); }
  Result operator()(const NamedEntity &x) const { return &x.GetLastSymbol(); }
  Result operator()(const ProcedureDesignator &x) const {
    return x.GetSymbol();
  }
  template <typename T> Result operator()(const Expr<T> &x) const {
    if constexpr (common::HasMember<T, AllIntrinsicTypes> ||
        std::is_same_v<T, SomeDerived>) {
      if (const auto *designator{std::get_if<Designator<T>>(&x.u)}) {
        if (auto known{(*this)(*designator)}) {
          return known;
        }
      }
      return nullptr;
    } else {
      return (*this)(x.u);
    }
  }
};

template <typename A> const Symbol *GetLastSymbol(const A &x) {
  if (auto known{GetLastSymbolHelper{}(x)}) {
    return *known;
  } else {
    return nullptr;
  }
}

// For everyday variables: if GetLastSymbol() succeeds on the argument, return
// its set of attributes, otherwise the empty set.  Also works on variables that
// are pointer results of functions.
template <typename A> semantics::Attrs GetAttrs(const A &x) {
  if (const Symbol * symbol{GetLastSymbol(x)}) {
    return symbol->attrs();
  } else {
    return {};
  }
}

template <>
inline semantics::Attrs GetAttrs<Expr<SomeType>>(const Expr<SomeType> &x) {
  if (IsVariable(x)) {
    if (const auto *procRef{UnwrapProcedureRef(x)}) {
      if (const Symbol * interface{procRef->proc().GetInterfaceSymbol()}) {
        if (const auto *details{
                interface->detailsIf<semantics::SubprogramDetails>()}) {
          if (details->isFunction() &&
              details->result().attrs().test(semantics::Attr::POINTER)) {
            // N.B.: POINTER becomes TARGET in SetAttrsFromAssociation()
            return details->result().attrs();
          }
        }
      }
    }
  }
  if (const Symbol * symbol{GetLastSymbol(x)}) {
    return symbol->attrs();
  } else {
    return {};
  }
}

template <typename A> semantics::Attrs GetAttrs(const std::optional<A> &x) {
  if (x) {
    return GetAttrs(*x);
  } else {
    return {};
  }
}

// GetBaseObject()
template <typename A> std::optional<BaseObject> GetBaseObject(const A &) {
  return std::nullopt;
}
template <typename T>
std::optional<BaseObject> GetBaseObject(const Designator<T> &x) {
  return x.GetBaseObject();
}
template <typename T>
std::optional<BaseObject> GetBaseObject(const Expr<T> &x) {
  return common::visit([](const auto &y) { return GetBaseObject(y); }, x.u);
}
template <typename A>
std::optional<BaseObject> GetBaseObject(const std::optional<A> &x) {
  if (x) {
    return GetBaseObject(*x);
  } else {
    return std::nullopt;
  }
}

// Like IsAllocatableOrPointer, but accepts pointer function results as being
// pointers too.
bool IsAllocatableOrPointerObject(const Expr<SomeType> &);

bool IsAllocatableDesignator(const Expr<SomeType> &);

// Procedure and pointer detection predicates
bool IsProcedureDesignator(const Expr<SomeType> &);
bool IsFunctionDesignator(const Expr<SomeType> &);
bool IsPointer(const Expr<SomeType> &);
bool IsProcedurePointer(const Expr<SomeType> &);
bool IsProcedure(const Expr<SomeType> &);
bool IsProcedurePointerTarget(const Expr<SomeType> &);
bool IsBareNullPointer(const Expr<SomeType> *); // NULL() w/o MOLD= or type
bool IsNullObjectPointer(const Expr<SomeType> &);
bool IsNullProcedurePointer(const Expr<SomeType> &);
bool IsNullPointer(const Expr<SomeType> &);
bool IsObjectPointer(const Expr<SomeType> &);

// Can Expr be passed as absent to an optional dummy argument.
// See 15.5.2.12 point 1 for more details.
bool MayBePassedAsAbsentOptional(const Expr<SomeType> &);

// Extracts the chain of symbols from a designator, which has perhaps been
// wrapped in an Expr<>, removing all of the (co)subscripts.  The
// base object will be the first symbol in the result vector.
struct GetSymbolVectorHelper
    : public Traverse<GetSymbolVectorHelper, SymbolVector> {
  using Result = SymbolVector;
  using Base = Traverse<GetSymbolVectorHelper, Result>;
  using Base::operator();
  GetSymbolVectorHelper() : Base{*this} {}
  Result Default() { return {}; }
  Result Combine(Result &&a, Result &&b) {
    a.insert(a.end(), b.begin(), b.end());
    return std::move(a);
  }
  Result operator()(const Symbol &) const;
  Result operator()(const Component &) const;
  Result operator()(const ArrayRef &) const;
  Result operator()(const CoarrayRef &) const;
};
template <typename A> SymbolVector GetSymbolVector(const A &x) {
  return GetSymbolVectorHelper{}(x);
}

// GetLastTarget() returns the rightmost symbol in an object designator's
// SymbolVector that has the POINTER or TARGET attribute, or a null pointer
// when none is found.
const Symbol *GetLastTarget(const SymbolVector &);

// Collects all of the Symbols in an expression
template <typename A> semantics::UnorderedSymbolSet CollectSymbols(const A &);
extern template semantics::UnorderedSymbolSet CollectSymbols(
    const Expr<SomeType> &);
extern template semantics::UnorderedSymbolSet CollectSymbols(
    const Expr<SomeInteger> &);
extern template semantics::UnorderedSymbolSet CollectSymbols(
    const Expr<SubscriptInteger> &);

// Collects Symbols of interest for the CUDA data transfer in an expression
template <typename A>
semantics::UnorderedSymbolSet CollectCudaSymbols(const A &);
extern template semantics::UnorderedSymbolSet CollectCudaSymbols(
    const Expr<SomeType> &);
extern template semantics::UnorderedSymbolSet CollectCudaSymbols(
    const Expr<SomeInteger> &);
extern template semantics::UnorderedSymbolSet CollectCudaSymbols(
    const Expr<SubscriptInteger> &);

// Predicate: does a variable contain a vector-valued subscript (not a triplet)?
bool HasVectorSubscript(const Expr<SomeType> &);

// Predicate: does an expression contain constant?
bool HasConstant(const Expr<SomeType> &);

// Utilities for attaching the location of the declaration of a symbol
// of interest to a message.  Handles the case of USE association gracefully.
parser::Message *AttachDeclaration(parser::Message &, const Symbol &);
parser::Message *AttachDeclaration(parser::Message *, const Symbol &);
template <typename MESSAGES, typename... A>
parser::Message *SayWithDeclaration(
    MESSAGES &messages, const Symbol &symbol, A &&...x) {
  return AttachDeclaration(messages.Say(std::forward<A>(x)...), symbol);
}

// Check for references to impure procedures; returns the name
// of one to complain about, if any exist.
std::optional<std::string> FindImpureCall(
    FoldingContext &, const Expr<SomeType> &);
std::optional<std::string> FindImpureCall(
    FoldingContext &, const ProcedureRef &);

// Predicate: is a scalar expression suitable for naive scalar expansion
// in the flattening of an array expression?
// TODO: capture such scalar expansions in temporaries, flatten everything
class UnexpandabilityFindingVisitor
    : public AnyTraverse<UnexpandabilityFindingVisitor> {
public:
  using Base = AnyTraverse<UnexpandabilityFindingVisitor>;
  using Base::operator();
  explicit UnexpandabilityFindingVisitor(bool admitPureCall)
      : Base{*this}, admitPureCall_{admitPureCall} {}
  template <typename T> bool operator()(const FunctionRef<T> &procRef) {
    return !admitPureCall_ || !procRef.proc().IsPure();
  }
  bool operator()(const CoarrayRef &) { return true; }

private:
  bool admitPureCall_{false};
};

template <typename T>
bool IsExpandableScalar(const Expr<T> &expr, FoldingContext &context,
    const Shape &shape, bool admitPureCall = false) {
  if (UnexpandabilityFindingVisitor{admitPureCall}(expr)) {
    auto extents{AsConstantExtents(context, shape)};
    return extents && GetSize(*extents) == 1;
  } else {
    return true;
  }
}

// Common handling for procedure pointer compatibility of left- and right-hand
// sides.  Returns nullopt if they're compatible.  Otherwise, it returns a
// message that needs to be augmented by the names of the left and right sides.
std::optional<parser::MessageFixedText> CheckProcCompatibility(bool isCall,
    const std::optional<characteristics::Procedure> &lhsProcedure,
    const characteristics::Procedure *rhsProcedure,
    const SpecificIntrinsic *specificIntrinsic, std::string &whyNotCompatible,
    std::optional<std::string> &warning, bool ignoreImplicitVsExplicit);

// Scalar constant expansion
class ScalarConstantExpander {
public:
  explicit ScalarConstantExpander(ConstantSubscripts &&extents)
      : extents_{std::move(extents)} {}
  ScalarConstantExpander(
      ConstantSubscripts &&extents, std::optional<ConstantSubscripts> &&lbounds)
      : extents_{std::move(extents)}, lbounds_{std::move(lbounds)} {}
  ScalarConstantExpander(
      ConstantSubscripts &&extents, ConstantSubscripts &&lbounds)
      : extents_{std::move(extents)}, lbounds_{std::move(lbounds)} {}

  template <typename A> A Expand(A &&x) const {
    return std::move(x); // default case
  }
  template <typename T> Constant<T> Expand(Constant<T> &&x) {
    auto expanded{x.Reshape(std::move(extents_))};
    if (lbounds_) {
      expanded.set_lbounds(std::move(*lbounds_));
    }
    return expanded;
  }
  template <typename T> Expr<T> Expand(Parentheses<T> &&x) {
    return Expand(std::move(x.left())); // Constant<> can be parenthesized
  }
  template <typename T> Expr<T> Expand(Expr<T> &&x) {
    return common::visit(
        [&](auto &&x) { return Expr<T>{Expand(std::move(x))}; },
        std::move(x.u));
  }

private:
  ConstantSubscripts extents_;
  std::optional<ConstantSubscripts> lbounds_;
};

// Given a collection of element values, package them as a Constant.
// If the type is Character or a derived type, take the length or type
// (resp.) from a another Constant.
template <typename T>
Constant<T> PackageConstant(std::vector<Scalar<T>> &&elements,
    const Constant<T> &reference, const ConstantSubscripts &shape) {
  if constexpr (T::category == TypeCategory::Character) {
    return Constant<T>{
        reference.LEN(), std::move(elements), ConstantSubscripts{shape}};
  } else if constexpr (T::category == TypeCategory::Derived) {
    return Constant<T>{reference.GetType().GetDerivedTypeSpec(),
        std::move(elements), ConstantSubscripts{shape}};
  } else {
    return Constant<T>{std::move(elements), ConstantSubscripts{shape}};
  }
}

// Nonstandard conversions of constants (integer->logical, logical->integer)
// that can appear in DATA statements as an extension.
std::optional<Expr<SomeType>> DataConstantConversionExtension(
    FoldingContext &, const DynamicType &, const Expr<SomeType> &);

// Convert Hollerith or short character to a another type as if the
// Hollerith data had been BOZ.
std::optional<Expr<SomeType>> HollerithToBOZ(
    FoldingContext &, const Expr<SomeType> &, const DynamicType &);

// Set explicit lower bounds on a constant array.
class ArrayConstantBoundChanger {
public:
  explicit ArrayConstantBoundChanger(ConstantSubscripts &&lbounds)
      : lbounds_{std::move(lbounds)} {}

  template <typename A> A ChangeLbounds(A &&x) const {
    return std::move(x); // default case
  }
  template <typename T> Constant<T> ChangeLbounds(Constant<T> &&x) {
    x.set_lbounds(std::move(lbounds_));
    return std::move(x);
  }
  template <typename T> Expr<T> ChangeLbounds(Parentheses<T> &&x) {
    return ChangeLbounds(
        std::move(x.left())); // Constant<> can be parenthesized
  }
  template <typename T> Expr<T> ChangeLbounds(Expr<T> &&x) {
    return common::visit(
        [&](auto &&x) { return Expr<T>{ChangeLbounds(std::move(x))}; },
        std::move(x.u)); // recurse until we hit a constant
  }

private:
  ConstantSubscripts &&lbounds_;
};

// Predicate: should two expressions be considered identical for the purposes
// of determining whether two procedure interfaces are compatible, modulo
// naming of corresponding dummy arguments?
template <typename T>
std::optional<bool> AreEquivalentInInterface(const Expr<T> &, const Expr<T> &);
extern template std::optional<bool> AreEquivalentInInterface<SubscriptInteger>(
    const Expr<SubscriptInteger> &, const Expr<SubscriptInteger> &);
extern template std::optional<bool> AreEquivalentInInterface<SomeInteger>(
    const Expr<SomeInteger> &, const Expr<SomeInteger> &);

bool CheckForCoindexedObject(parser::ContextualMessages &,
    const std::optional<ActualArgument> &, const std::string &procName,
    const std::string &argName);

inline bool CanCUDASymbolHaveSaveAttr(const Symbol &sym) {
  if (const auto *details =
          sym.GetUltimate().detailsIf<semantics::ObjectEntityDetails>()) {
    if (details->cudaDataAttr() &&
        *details->cudaDataAttr() != common::CUDADataAttr::Unified) {
      return false;
    }
  }
  return true;
}

inline bool IsCUDADeviceSymbol(const Symbol &sym) {
  if (const auto *details =
          sym.GetUltimate().detailsIf<semantics::ObjectEntityDetails>()) {
    if (details->cudaDataAttr() &&
        *details->cudaDataAttr() != common::CUDADataAttr::Pinned) {
      return true;
    }
  }
  return false;
}

// Get the number of distinct symbols with CUDA device
// attribute in the expression.
template <typename A> inline int GetNbOfCUDADeviceSymbols(const A &expr) {
  semantics::UnorderedSymbolSet symbols;
  for (const Symbol &sym : CollectCudaSymbols(expr)) {
    if (IsCUDADeviceSymbol(sym)) {
      symbols.insert(sym);
    }
  }
  return symbols.size();
}

// Check if any of the symbols part of the expression has a CUDA device
// attribute.
template <typename A> inline bool HasCUDADeviceAttrs(const A &expr) {
  return GetNbOfCUDADeviceSymbols(expr) > 0;
}

/// Check if the expression is a mix of host and device variables that require
/// implicit data transfer.
inline bool HasCUDAImplicitTransfer(const Expr<SomeType> &expr) {
  unsigned hostSymbols{0};
  unsigned deviceSymbols{0};
  for (const Symbol &sym : CollectCudaSymbols(expr)) {
    if (IsCUDADeviceSymbol(sym)) {
      ++deviceSymbols;
    } else {
      if (sym.owner().IsDerivedType()) {
        if (IsCUDADeviceSymbol(sym.owner().GetSymbol()->GetUltimate())) {
          ++deviceSymbols;
        }
      }
      ++hostSymbols;
    }
  }
  bool hasConstant{HasConstant(expr)};
  return (hasConstant || (hostSymbols > 0)) && deviceSymbols > 0;
}

} // namespace Fortran::evaluate

namespace Fortran::semantics {

class Scope;

// If a symbol represents an ENTRY, return the symbol of the main entry
// point to its subprogram.
const Symbol *GetMainEntry(const Symbol *);

// These functions are used in Evaluate so they are defined here rather than in
// Semantics to avoid a link-time dependency on Semantics.
// All of these apply GetUltimate() or ResolveAssociations() to their arguments.
bool IsVariableName(const Symbol &);
bool IsPureProcedure(const Symbol &);
bool IsPureProcedure(const Scope &);
bool IsExplicitlyImpureProcedure(const Symbol &);
bool IsElementalProcedure(const Symbol &);
bool IsFunction(const Symbol &);
bool IsFunction(const Scope &);
bool IsProcedure(const Symbol &);
bool IsProcedure(const Scope &);
bool IsProcedurePointer(const Symbol *);
bool IsProcedurePointer(const Symbol &);
bool IsObjectPointer(const Symbol *);
bool IsAllocatableOrObjectPointer(const Symbol *);
bool IsAutomatic(const Symbol &);
bool IsSaved(const Symbol &); // saved implicitly or explicitly
bool IsDummy(const Symbol &);
bool IsAssumedShape(const Symbol &);
bool IsDeferredShape(const Symbol &);
bool IsFunctionResult(const Symbol &);
bool IsKindTypeParameter(const Symbol &);
bool IsLenTypeParameter(const Symbol &);
bool IsExtensibleType(const DerivedTypeSpec *);
bool IsSequenceOrBindCType(const DerivedTypeSpec *);
bool IsBuiltinDerivedType(const DerivedTypeSpec *derived, const char *name);
bool IsBuiltinCPtr(const Symbol &);
bool IsEventType(const DerivedTypeSpec *);
bool IsLockType(const DerivedTypeSpec *);
bool IsNotifyType(const DerivedTypeSpec *);
// Is this derived type IEEE_FLAG_TYPE from module ISO_IEEE_EXCEPTIONS?
bool IsIeeeFlagType(const DerivedTypeSpec *);
// Is this derived type IEEE_ROUND_TYPE from module ISO_IEEE_ARITHMETIC?
bool IsIeeeRoundType(const DerivedTypeSpec *);
// Is this derived type TEAM_TYPE from module ISO_FORTRAN_ENV?
bool IsTeamType(const DerivedTypeSpec *);
// Is this derived type TEAM_TYPE, C_PTR, or C_FUNPTR?
bool IsBadCoarrayType(const DerivedTypeSpec *);
// Is this derived type either C_PTR or C_FUNPTR from module ISO_C_BINDING
bool IsIsoCType(const DerivedTypeSpec *);
bool IsEventTypeOrLockType(const DerivedTypeSpec *);
inline bool IsAssumedSizeArray(const Symbol &symbol) {
  if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
    return (object->isDummy() || symbol.test(Symbol::Flag::CrayPointee)) &&
        object->shape().CanBeAssumedSize();
  } else if (const auto *assoc{symbol.detailsIf<AssocEntityDetails>()}) {
    return assoc->IsAssumedSize();
  } else {
    return false;
  }
}

// ResolveAssociations() traverses use associations and host associations
// like GetUltimate(), but also resolves through whole variable associations
// with ASSOCIATE(x => y) and related constructs.  GetAssociationRoot()
// applies ResolveAssociations() and then, in the case of resolution to
// a construct association with part of a variable that does not involve a
// vector subscript, returns the first symbol of that variable instead
// of the construct entity.
// (E.g., for ASSOCIATE(x => y%z), ResolveAssociations(x) returns x,
// while GetAssociationRoot(x) returns y.)
// In a SELECT RANK construct, ResolveAssociations() stops at a
// RANK(n) or RANK(*) case symbol, but traverses the selector for
// RANK DEFAULT.
const Symbol &ResolveAssociations(const Symbol &);
const Symbol &GetAssociationRoot(const Symbol &);

const Symbol *FindCommonBlockContaining(const Symbol &);
int CountLenParameters(const DerivedTypeSpec &);
int CountNonConstantLenParameters(const DerivedTypeSpec &);

const Symbol &GetUsedModule(const UseDetails &);
const Symbol *FindFunctionResult(const Symbol &);

// Type compatibility predicate: are x and y effectively the same type?
// Uses DynamicType::IsTkCompatible(), which handles the case of distinct
// but identical derived types.
bool AreTkCompatibleTypes(const DeclTypeSpec *x, const DeclTypeSpec *y);

common::IgnoreTKRSet GetIgnoreTKR(const Symbol &);

std::optional<int> GetDummyArgumentNumber(const Symbol *);

const Symbol *FindAncestorModuleProcedure(const Symbol *symInSubmodule);

} // namespace Fortran::semantics

#endif // FORTRAN_EVALUATE_TOOLS_H_
