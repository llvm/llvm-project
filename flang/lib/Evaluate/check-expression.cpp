//===-- lib/Evaluate/check-expression.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/check-expression.h"
#include "flang/Evaluate/characteristics.h"
#include "flang/Evaluate/intrinsics.h"
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Evaluate/type.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include <set>
#include <string>

namespace Fortran::evaluate {

// Constant expression predicates IsConstantExpr() & IsScopeInvariantExpr().
// This code determines whether an expression is a "constant expression"
// in the sense of section 10.1.12.  This is not the same thing as being
// able to fold it (yet) into a known constant value; specifically,
// the expression may reference derived type kind parameters whose values
// are not yet known.
//
// The variant form (IsScopeInvariantExpr()) also accepts symbols that are
// INTENT(IN) dummy arguments without the VALUE attribute.
template <bool INVARIANT>
class IsConstantExprHelper
    : public AllTraverse<IsConstantExprHelper<INVARIANT>, true> {
public:
  using Base = AllTraverse<IsConstantExprHelper, true>;
  IsConstantExprHelper() : Base{*this} {}
  using Base::operator();

  // A missing expression is not considered to be constant.
  template <typename A> bool operator()(const std::optional<A> &x) const {
    return x && (*this)(*x);
  }

  bool operator()(const TypeParamInquiry &inq) const {
    return INVARIANT || semantics::IsKindTypeParameter(inq.parameter());
  }
  bool operator()(const semantics::Symbol &symbol) const {
    const auto &ultimate{GetAssociationRoot(symbol)};
    return IsNamedConstant(ultimate) || IsImpliedDoIndex(ultimate) ||
        IsInitialProcedureTarget(ultimate) ||
        ultimate.has<semantics::TypeParamDetails>() ||
        (INVARIANT && IsIntentIn(symbol) && !IsOptional(symbol) &&
            !symbol.attrs().test(semantics::Attr::VALUE));
  }
  bool operator()(const CoarrayRef &) const { return false; }
  bool operator()(const semantics::ParamValue &param) const {
    return param.isExplicit() && (*this)(param.GetExplicit());
  }
  bool operator()(const ProcedureRef &) const;
  bool operator()(const StructureConstructor &constructor) const {
    for (const auto &[symRef, expr] : constructor) {
      if (!IsConstantStructureConstructorComponent(*symRef, expr.value())) {
        return false;
      }
    }
    return true;
  }
  bool operator()(const Component &component) const {
    return (*this)(component.base());
  }
  // Prevent integer division by known zeroes in constant expressions.
  template <int KIND>
  bool operator()(
      const Divide<Type<TypeCategory::Integer, KIND>> &division) const {
    using T = Type<TypeCategory::Integer, KIND>;
    if ((*this)(division.left()) && (*this)(division.right())) {
      const auto divisor{GetScalarConstantValue<T>(division.right())};
      return !divisor || !divisor->IsZero();
    } else {
      return false;
    }
  }

  bool operator()(const Constant<SomeDerived> &) const { return true; }
  bool operator()(const DescriptorInquiry &x) const {
    const Symbol &sym{x.base().GetLastSymbol()};
    return INVARIANT && !IsAllocatable(sym) &&
        (!IsDummy(sym) ||
            (IsIntentIn(sym) && !IsOptional(sym) &&
                !sym.attrs().test(semantics::Attr::VALUE)));
  }

private:
  bool IsConstantStructureConstructorComponent(
      const Symbol &, const Expr<SomeType> &) const;
  bool IsConstantExprShape(const Shape &) const;
};

template <bool INVARIANT>
bool IsConstantExprHelper<INVARIANT>::IsConstantStructureConstructorComponent(
    const Symbol &component, const Expr<SomeType> &expr) const {
  if (IsAllocatable(component)) {
    return IsNullObjectPointer(&expr);
  } else if (IsPointer(component)) {
    return IsNullPointerOrAllocatable(&expr) || IsInitialDataTarget(expr) ||
        IsInitialProcedureTarget(expr);
  } else {
    return (*this)(expr);
  }
}

template <bool INVARIANT>
bool IsConstantExprHelper<INVARIANT>::operator()(
    const ProcedureRef &call) const {
  // LBOUND, UBOUND, and SIZE with truly constant DIM= arguments will have
  // been rewritten into DescriptorInquiry operations.
  if (const auto *intrinsic{std::get_if<SpecificIntrinsic>(&call.proc().u)}) {
    const characteristics::Procedure &proc{intrinsic->characteristics.value()};
    if (intrinsic->name == "kind" ||
        intrinsic->name == IntrinsicProcTable::InvalidName ||
        call.arguments().empty() || !call.arguments()[0]) {
      // kind is always a constant, and we avoid cascading errors by considering
      // invalid calls to intrinsics to be constant
      return true;
    } else if (intrinsic->name == "lbound") {
      auto base{ExtractNamedEntity(call.arguments()[0]->UnwrapExpr())};
      return base && IsConstantExprShape(GetLBOUNDs(*base));
    } else if (intrinsic->name == "ubound") {
      auto base{ExtractNamedEntity(call.arguments()[0]->UnwrapExpr())};
      return base && IsConstantExprShape(GetUBOUNDs(*base));
    } else if (intrinsic->name == "shape" || intrinsic->name == "size") {
      auto shape{GetShape(call.arguments()[0]->UnwrapExpr())};
      return shape && IsConstantExprShape(*shape);
    } else if (proc.IsPure()) {
      std::size_t j{0};
      for (const auto &arg : call.arguments()) {
        const auto *dataDummy{j < proc.dummyArguments.size()
                ? std::get_if<characteristics::DummyDataObject>(
                      &proc.dummyArguments[j].u)
                : nullptr};
        if (dataDummy &&
            dataDummy->attrs.test(
                characteristics::DummyDataObject::Attr::OnlyIntrinsicInquiry)) {
          // The value of the argument doesn't matter
        } else if (!arg) {
          if (dataDummy &&
              dataDummy->attrs.test(
                  characteristics::DummyDataObject::Attr::Optional)) {
            // Missing optional arguments are okay.
          } else {
            return false;
          }
        } else if (const auto *expr{arg->UnwrapExpr()};
            !expr || !(*this)(*expr)) {
          return false;
        }
        ++j;
      }
      return true;
    }
    // TODO: STORAGE_SIZE
  }
  return false;
}

template <bool INVARIANT>
bool IsConstantExprHelper<INVARIANT>::IsConstantExprShape(
    const Shape &shape) const {
  for (const auto &extent : shape) {
    if (!(*this)(extent)) {
      return false;
    }
  }
  return true;
}

template <typename A> bool IsConstantExpr(const A &x) {
  return IsConstantExprHelper<false>{}(x);
}
template bool IsConstantExpr(const Expr<SomeType> &);
template bool IsConstantExpr(const Expr<SomeInteger> &);
template bool IsConstantExpr(const Expr<SubscriptInteger> &);
template bool IsConstantExpr(const StructureConstructor &);

// IsScopeInvariantExpr()
template <typename A> bool IsScopeInvariantExpr(const A &x) {
  return IsConstantExprHelper<true>{}(x);
}
template bool IsScopeInvariantExpr(const Expr<SomeType> &);
template bool IsScopeInvariantExpr(const Expr<SomeInteger> &);
template bool IsScopeInvariantExpr(const Expr<SubscriptInteger> &);

// IsActuallyConstant()
struct IsActuallyConstantHelper {
  template <typename A> bool operator()(const A &) { return false; }
  template <typename T> bool operator()(const Constant<T> &) { return true; }
  template <typename T> bool operator()(const Parentheses<T> &x) {
    return (*this)(x.left());
  }
  template <typename T> bool operator()(const Expr<T> &x) {
    return common::visit([=](const auto &y) { return (*this)(y); }, x.u);
  }
  bool operator()(const Expr<SomeType> &x) {
    return common::visit([this](const auto &y) { return (*this)(y); }, x.u);
  }
  bool operator()(const StructureConstructor &x) {
    for (const auto &pair : x) {
      const Expr<SomeType> &y{pair.second.value()};
      const auto sym{pair.first};
      const bool compIsConstant{(*this)(y)};
      // If an allocatable component is initialized by a constant,
      // the structure constructor is not a constant.
      if ((!compIsConstant && !IsNullPointerOrAllocatable(&y)) ||
          (compIsConstant && IsAllocatable(sym))) {
        return false;
      }
    }
    return true;
  }
  template <typename A> bool operator()(const A *x) { return x && (*this)(*x); }
  template <typename A> bool operator()(const std::optional<A> &x) {
    return x && (*this)(*x);
  }
};

template <typename A> bool IsActuallyConstant(const A &x) {
  return IsActuallyConstantHelper{}(x);
}

template bool IsActuallyConstant(const Expr<SomeType> &);
template bool IsActuallyConstant(const Expr<SomeInteger> &);
template bool IsActuallyConstant(const Expr<SubscriptInteger> &);
template bool IsActuallyConstant(const std::optional<Expr<SubscriptInteger>> &);

// Object pointer initialization checking predicate IsInitialDataTarget().
// This code determines whether an expression is allowable as the static
// data address used to initialize a pointer with "=> x".  See C765.
class IsInitialDataTargetHelper
    : public AllTraverse<IsInitialDataTargetHelper, true> {
public:
  using Base = AllTraverse<IsInitialDataTargetHelper, true>;
  using Base::operator();
  explicit IsInitialDataTargetHelper(parser::ContextualMessages *m)
      : Base{*this}, messages_{m} {}

  bool emittedMessage() const { return emittedMessage_; }

  bool operator()(const BOZLiteralConstant &) const { return false; }
  bool operator()(const NullPointer &) const { return true; }
  template <typename T> bool operator()(const Constant<T> &) const {
    return false;
  }
  bool operator()(const semantics::Symbol &symbol) {
    // This function checks only base symbols, not components.
    const Symbol &ultimate{symbol.GetUltimate()};
    if (const auto *assoc{
            ultimate.detailsIf<semantics::AssocEntityDetails>()}) {
      if (const auto &expr{assoc->expr()}) {
        if (IsVariable(*expr)) {
          return (*this)(*expr);
        } else if (messages_) {
          messages_->Say(
              "An initial data target may not be an associated expression ('%s')"_err_en_US,
              ultimate.name());
          emittedMessage_ = true;
        }
      }
      return false;
    } else if (!CheckVarOrComponent(ultimate)) {
      return false;
    } else if (!ultimate.attrs().test(semantics::Attr::TARGET)) {
      if (messages_) {
        messages_->Say(
            "An initial data target may not be a reference to an object '%s' that lacks the TARGET attribute"_err_en_US,
            ultimate.name());
        emittedMessage_ = true;
      }
      return false;
    } else if (!IsSaved(ultimate)) {
      if (messages_) {
        messages_->Say(
            "An initial data target may not be a reference to an object '%s' that lacks the SAVE attribute"_err_en_US,
            ultimate.name());
        emittedMessage_ = true;
      }
      return false;
    } else {
      return true;
    }
  }
  bool operator()(const StaticDataObject &) const { return false; }
  bool operator()(const TypeParamInquiry &) const { return false; }
  bool operator()(const Triplet &x) const {
    return IsConstantExpr(x.lower()) && IsConstantExpr(x.upper()) &&
        IsConstantExpr(x.stride());
  }
  bool operator()(const Subscript &x) const {
    return common::visit(common::visitors{
                             [&](const Triplet &t) { return (*this)(t); },
                             [&](const auto &y) {
                               return y.value().Rank() == 0 &&
                                   IsConstantExpr(y.value());
                             },
                         },
        x.u);
  }
  bool operator()(const CoarrayRef &) const { return false; }
  bool operator()(const Component &x) {
    return CheckVarOrComponent(x.GetLastSymbol()) && (*this)(x.base());
  }
  bool operator()(const Substring &x) const {
    return IsConstantExpr(x.lower()) && IsConstantExpr(x.upper()) &&
        (*this)(x.parent());
  }
  bool operator()(const DescriptorInquiry &) const { return false; }
  template <typename T> bool operator()(const ArrayConstructor<T> &) const {
    return false;
  }
  bool operator()(const StructureConstructor &) const { return false; }
  template <typename D, typename R, typename... O>
  bool operator()(const Operation<D, R, O...> &) const {
    return false;
  }
  template <typename T> bool operator()(const Parentheses<T> &x) const {
    return (*this)(x.left());
  }
  bool operator()(const ProcedureRef &x) const {
    if (const SpecificIntrinsic * intrinsic{x.proc().GetSpecificIntrinsic()}) {
      return intrinsic->characteristics.value().attrs.test(
                 characteristics::Procedure::Attr::NullPointer) ||
          intrinsic->characteristics.value().attrs.test(
              characteristics::Procedure::Attr::NullAllocatable);
    }
    return false;
  }
  bool operator()(const Relational<SomeType> &) const { return false; }

private:
  bool CheckVarOrComponent(const semantics::Symbol &symbol) {
    const Symbol &ultimate{symbol.GetUltimate()};
    const char *unacceptable{nullptr};
    if (ultimate.Corank() > 0) {
      unacceptable = "a coarray";
    } else if (IsAllocatable(ultimate)) {
      unacceptable = "an ALLOCATABLE";
    } else if (IsPointer(ultimate)) {
      unacceptable = "a POINTER";
    } else {
      return true;
    }
    if (messages_) {
      messages_->Say(
          "An initial data target may not be a reference to %s '%s'"_err_en_US,
          unacceptable, ultimate.name());
      emittedMessage_ = true;
    }
    return false;
  }

  parser::ContextualMessages *messages_;
  bool emittedMessage_{false};
};

bool IsInitialDataTarget(
    const Expr<SomeType> &x, parser::ContextualMessages *messages) {
  IsInitialDataTargetHelper helper{messages};
  bool result{helper(x)};
  if (!result && messages && !helper.emittedMessage()) {
    messages->Say(
        "An initial data target must be a designator with constant subscripts"_err_en_US);
  }
  return result;
}

bool IsInitialProcedureTarget(const semantics::Symbol &symbol) {
  const auto &ultimate{symbol.GetUltimate()};
  return common::visit(
      common::visitors{
          [&](const semantics::SubprogramDetails &subp) {
            return !subp.isDummy() && !subp.stmtFunction() &&
                symbol.owner().kind() != semantics::Scope::Kind::MainProgram &&
                symbol.owner().kind() != semantics::Scope::Kind::Subprogram;
          },
          [](const semantics::SubprogramNameDetails &x) {
            return x.kind() != semantics::SubprogramKind::Internal;
          },
          [&](const semantics::ProcEntityDetails &proc) {
            return !semantics::IsPointer(ultimate) && !proc.isDummy();
          },
          [](const auto &) { return false; },
      },
      ultimate.details());
}

bool IsInitialProcedureTarget(const ProcedureDesignator &proc) {
  if (const auto *intrin{proc.GetSpecificIntrinsic()}) {
    return !intrin->isRestrictedSpecific;
  } else if (proc.GetComponent()) {
    return false;
  } else {
    return IsInitialProcedureTarget(DEREF(proc.GetSymbol()));
  }
}

bool IsInitialProcedureTarget(const Expr<SomeType> &expr) {
  if (const auto *proc{std::get_if<ProcedureDesignator>(&expr.u)}) {
    return IsInitialProcedureTarget(*proc);
  } else {
    return IsNullProcedurePointer(&expr);
  }
}

class SuspiciousRealLiteralFinder
    : public AnyTraverse<SuspiciousRealLiteralFinder> {
public:
  using Base = AnyTraverse<SuspiciousRealLiteralFinder>;
  SuspiciousRealLiteralFinder(int kind, FoldingContext &c)
      : Base{*this}, kind_{kind}, context_{c} {}
  using Base::operator();
  template <int KIND>
  bool operator()(const Constant<Type<TypeCategory::Real, KIND>> &x) const {
    if (kind_ > KIND && x.result().isFromInexactLiteralConversion()) {
      context_.Warn(common::UsageWarning::RealConstantWidening,
          "Default real literal in REAL(%d) context might need a kind suffix, as its rounded value %s is inexact"_warn_en_US,
          kind_, x.AsFortran());
      return true;
    } else {
      return false;
    }
  }
  template <int KIND>
  bool operator()(const Constant<Type<TypeCategory::Complex, KIND>> &x) const {
    if (kind_ > KIND && x.result().isFromInexactLiteralConversion()) {
      context_.Warn(common::UsageWarning::RealConstantWidening,
          "Default real literal in COMPLEX(%d) context might need a kind suffix, as its rounded value %s is inexact"_warn_en_US,
          kind_, x.AsFortran());
      return true;
    } else {
      return false;
    }
  }
  template <TypeCategory TOCAT, int TOKIND, TypeCategory FROMCAT>
  bool operator()(const Convert<Type<TOCAT, TOKIND>, FROMCAT> &x) const {
    if constexpr ((TOCAT == TypeCategory::Real ||
                      TOCAT == TypeCategory::Complex) &&
        (FROMCAT == TypeCategory::Real || FROMCAT == TypeCategory::Complex)) {
      auto fromType{x.left().GetType()};
      if (!fromType || fromType->kind() < TOKIND) {
        return false;
      }
    }
    return (*this)(x.left());
  }

private:
  int kind_;
  FoldingContext &context_;
};

void CheckRealWidening(const Expr<SomeType> &expr, const DynamicType &toType,
    FoldingContext &context) {
  if (toType.category() == TypeCategory::Real ||
      toType.category() == TypeCategory::Complex) {
    if (auto fromType{expr.GetType()}) {
      if ((fromType->category() == TypeCategory::Real ||
              fromType->category() == TypeCategory::Complex) &&
          toType.kind() > fromType->kind()) {
        SuspiciousRealLiteralFinder{toType.kind(), context}(expr);
      }
    }
  }
}

void CheckRealWidening(const Expr<SomeType> &expr,
    const std::optional<DynamicType> &toType, FoldingContext &context) {
  if (toType) {
    CheckRealWidening(expr, *toType, context);
  }
}

class InexactLiteralConversionFlagClearer
    : public AnyTraverse<InexactLiteralConversionFlagClearer> {
public:
  using Base = AnyTraverse<InexactLiteralConversionFlagClearer>;
  InexactLiteralConversionFlagClearer() : Base(*this) {}
  using Base::operator();
  template <int KIND>
  bool operator()(const Constant<Type<TypeCategory::Real, KIND>> &x) const {
    auto &mut{const_cast<Type<TypeCategory::Real, KIND> &>(x.result())};
    mut.set_isFromInexactLiteralConversion(false);
    return false;
  }
};

// Converts, folds, and then checks type, rank, and shape of an
// initialization expression for a named constant, a non-pointer
// variable static initialization, a component default initializer,
// a type parameter default value, or instantiated type parameter value.
std::optional<Expr<SomeType>> NonPointerInitializationExpr(const Symbol &symbol,
    Expr<SomeType> &&x, FoldingContext &context,
    const semantics::Scope *instantiation) {
  CHECK(!IsPointer(symbol));
  if (auto symTS{
          characteristics::TypeAndShape::Characterize(symbol, context)}) {
    auto xType{x.GetType()};
    CheckRealWidening(x, symTS->type(), context);
    auto converted{ConvertToType(symTS->type(), Expr<SomeType>{x})};
    if (!converted &&
        symbol.owner().context().IsEnabled(
            common::LanguageFeature::LogicalIntegerAssignment)) {
      converted = DataConstantConversionExtension(context, symTS->type(), x);
      if (converted) {
        context.Warn(common::LanguageFeature::LogicalIntegerAssignment,
            "nonstandard usage: initialization of %s with %s"_port_en_US,
            symTS->type().AsFortran(), x.GetType().value().AsFortran());
      }
    }
    if (converted) {
      auto folded{Fold(context, std::move(*converted))};
      if (IsActuallyConstant(folded)) {
        InexactLiteralConversionFlagClearer{}(folded);
        int symRank{symTS->Rank()};
        if (IsImpliedShape(symbol)) {
          if (folded.Rank() == symRank) {
            return ArrayConstantBoundChanger{
                std::move(*AsConstantExtents(
                    context, GetRawLowerBounds(context, NamedEntity{symbol})))}
                .ChangeLbounds(std::move(folded));
          } else {
            context.messages().Say(
                "Implied-shape parameter '%s' has rank %d but its initializer has rank %d"_err_en_US,
                symbol.name(), symRank, folded.Rank());
          }
        } else if (auto extents{AsConstantExtents(context, symTS->shape())};
            extents && !HasNegativeExtent(*extents)) {
          if (folded.Rank() == 0 && symRank == 0) {
            // symbol and constant are both scalars
            return {std::move(folded)};
          } else if (folded.Rank() == 0 && symRank > 0) {
            // expand the scalar constant to an array
            return ScalarConstantExpander{std::move(*extents),
                AsConstantExtents(
                    context, GetRawLowerBounds(context, NamedEntity{symbol}))}
                .Expand(std::move(folded));
          } else if (auto resultShape{GetShape(context, folded)}) {
            CHECK(symTS->shape()); // Assumed-ranks cannot be initialized.
            if (CheckConformance(context.messages(), *symTS->shape(),
                    *resultShape, CheckConformanceFlags::None,
                    "initialized object", "initialization expression")
                    .value_or(false /*fail if not known now to conform*/)) {
              // make a constant array with adjusted lower bounds
              return ArrayConstantBoundChanger{
                  std::move(*AsConstantExtents(context,
                      GetRawLowerBounds(context, NamedEntity{symbol})))}
                  .ChangeLbounds(std::move(folded));
            }
          }
        } else if (IsNamedConstant(symbol)) {
          if (IsExplicitShape(symbol)) {
            context.messages().Say(
                "Named constant '%s' array must have constant shape"_err_en_US,
                symbol.name());
          } else {
            // Declaration checking handles other cases
          }
        } else {
          context.messages().Say(
              "Shape of initialized object '%s' must be constant"_err_en_US,
              symbol.name());
        }
      } else if (IsErrorExpr(folded)) {
      } else if (IsLenTypeParameter(symbol)) {
        return {std::move(folded)};
      } else if (IsKindTypeParameter(symbol)) {
        if (instantiation) {
          context.messages().Say(
              "Value of kind type parameter '%s' (%s) must be a scalar INTEGER constant"_err_en_US,
              symbol.name(), folded.AsFortran());
        } else {
          return {std::move(folded)};
        }
      } else if (IsNamedConstant(symbol)) {
        if (symbol.name() == "numeric_storage_size" &&
            symbol.owner().IsModule() &&
            DEREF(symbol.owner().symbol()).name() == "iso_fortran_env") {
          // Very special case: numeric_storage_size is not folded until
          // it read from the iso_fortran_env module file, as its value
          // depends on compilation options.
          return {std::move(folded)};
        }
        context.messages().Say(
            "Value of named constant '%s' (%s) cannot be computed as a constant value"_err_en_US,
            symbol.name(), folded.AsFortran());
      } else {
        context.messages().Say(
            "Initialization expression for '%s' (%s) cannot be computed as a constant value"_err_en_US,
            symbol.name(), x.AsFortran());
      }
    } else if (xType) {
      context.messages().Say(
          "Initialization expression cannot be converted to declared type of '%s' from %s"_err_en_US,
          symbol.name(), xType->AsFortran());
    } else {
      context.messages().Say(
          "Initialization expression cannot be converted to declared type of '%s'"_err_en_US,
          symbol.name());
    }
  }
  return std::nullopt;
}

// Specification expression validation (10.1.11(2), C1010)
class CheckSpecificationExprHelper
    : public AnyTraverse<CheckSpecificationExprHelper,
          std::optional<std::string>> {
public:
  using Result = std::optional<std::string>;
  using Base = AnyTraverse<CheckSpecificationExprHelper, Result>;
  explicit CheckSpecificationExprHelper(const semantics::Scope &s,
      FoldingContext &context, bool forElementalFunctionResult)
      : Base{*this}, scope_{s}, context_{context},
        forElementalFunctionResult_{forElementalFunctionResult} {}
  using Base::operator();

  Result operator()(const CoarrayRef &) const { return "coindexed reference"; }

  Result operator()(const semantics::Symbol &symbol) const {
    const auto &ultimate{symbol.GetUltimate()};
    const auto *object{ultimate.detailsIf<semantics::ObjectEntityDetails>()};
    bool isInitialized{semantics::IsSaved(ultimate) &&
        !IsAllocatable(ultimate) && object &&
        (ultimate.test(Symbol::Flag::InDataStmt) ||
            object->init().has_value())};
    bool hasHostAssociation{
        &symbol.owner() != &scope_ || &ultimate.owner() != &scope_};
    if (const auto *assoc{
            ultimate.detailsIf<semantics::AssocEntityDetails>()}) {
      return (*this)(assoc->expr());
    } else if (semantics::IsNamedConstant(ultimate) ||
        ultimate.owner().IsModule() || ultimate.owner().IsSubmodule()) {
      return std::nullopt;
    } else if (scope_.IsDerivedType() &&
        IsVariableName(ultimate)) { // C750, C754
      return "derived type component or type parameter value not allowed to "
             "reference variable '"s +
          ultimate.name().ToString() + "'";
    } else if (IsDummy(ultimate)) {
      if (!inInquiry_ && forElementalFunctionResult_) {
        return "dependence on value of dummy argument '"s +
            ultimate.name().ToString() + "'";
      } else if (ultimate.attrs().test(semantics::Attr::OPTIONAL)) {
        return "reference to OPTIONAL dummy argument '"s +
            ultimate.name().ToString() + "'";
      } else if (!inInquiry_ && !hasHostAssociation &&
          ultimate.attrs().test(semantics::Attr::INTENT_OUT)) {
        return "reference to INTENT(OUT) dummy argument '"s +
            ultimate.name().ToString() + "'";
      } else if (!ultimate.has<semantics::ObjectEntityDetails>()) {
        return "dummy procedure argument";
      } else {
        // Sketchy case: some compilers allow an INTENT(OUT) dummy argument
        // to be used in a specification expression if it is host-associated.
        // The arguments raised in support this usage, however, depend on
        // a reading of the standard that would also accept an OPTIONAL
        // host-associated dummy argument, and that doesn't seem like a
        // good idea.
        if (!inInquiry_ && hasHostAssociation &&
            ultimate.attrs().test(semantics::Attr::INTENT_OUT)) {
          context_.Warn(common::UsageWarning::HostAssociatedIntentOutInSpecExpr,
              "specification expression refers to host-associated INTENT(OUT) dummy argument '%s'"_port_en_US,
              ultimate.name());
        }
        return std::nullopt;
      }
    } else if (hasHostAssociation) {
      return std::nullopt; // host association is in play
    } else if (isInitialized &&
        context_.languageFeatures().IsEnabled(
            common::LanguageFeature::SavedLocalInSpecExpr)) {
      context_.Warn(common::LanguageFeature::SavedLocalInSpecExpr,
          "specification expression refers to local object '%s' (initialized and saved)"_port_en_US,
          ultimate.name());
      return std::nullopt;
    } else if (const auto *object{
                   ultimate.detailsIf<semantics::ObjectEntityDetails>()}) {
      if (object->commonBlock()) {
        return std::nullopt;
      }
    }
    if (inInquiry_) {
      return std::nullopt;
    } else {
      return "reference to local entity '"s + ultimate.name().ToString() + "'";
    }
  }

  Result operator()(const Component &x) const {
    // Don't look at the component symbol.
    return (*this)(x.base());
  }
  Result operator()(const ArrayRef &x) const {
    if (auto result{(*this)(x.base())}) {
      return result;
    }
    // The subscripts don't get special protection for being in a
    // specification inquiry context;
    auto restorer{common::ScopedSet(inInquiry_, false)};
    return (*this)(x.subscript());
  }
  Result operator()(const Substring &x) const {
    if (auto result{(*this)(x.parent())}) {
      return result;
    }
    // The bounds don't get special protection for being in a
    // specification inquiry context;
    auto restorer{common::ScopedSet(inInquiry_, false)};
    if (auto result{(*this)(x.lower())}) {
      return result;
    }
    return (*this)(x.upper());
  }
  Result operator()(const DescriptorInquiry &x) const {
    // Many uses of SIZE(), LBOUND(), &c. that are valid in specification
    // expressions will have been converted to expressions over descriptor
    // inquiries by Fold().
    // Catch REAL, ALLOCATABLE :: X(:); REAL :: Y(SIZE(X))
    if (IsPermissibleInquiry(
            x.base().GetFirstSymbol(), x.base().GetLastSymbol(), x.field())) {
      auto restorer{common::ScopedSet(inInquiry_, true)};
      return (*this)(x.base());
    } else if (IsConstantExpr(x)) {
      return std::nullopt;
    } else {
      return "non-constant descriptor inquiry not allowed for local object";
    }
  }

  Result operator()(const TypeParamInquiry &inq) const {
    if (scope_.IsDerivedType()) {
      if (!IsConstantExpr(inq) &&
          inq.base() /* X%T, not local T */) { // C750, C754
        return "non-constant reference to a type parameter inquiry not allowed "
               "for derived type components or type parameter values";
      }
    } else if (inq.base() &&
        IsInquiryAlwaysPermissible(inq.base()->GetFirstSymbol())) {
      auto restorer{common::ScopedSet(inInquiry_, true)};
      return (*this)(inq.base());
    } else if (!IsConstantExpr(inq)) {
      return "non-constant type parameter inquiry not allowed for local object";
    }
    return std::nullopt;
  }

  Result operator()(const ProcedureRef &x) const {
    if (const auto *symbol{x.proc().GetSymbol()}) {
      const Symbol &ultimate{symbol->GetUltimate()};
      if (!semantics::IsPureProcedure(ultimate)) {
        return "reference to impure function '"s + ultimate.name().ToString() +
            "'";
      }
      if (semantics::IsStmtFunction(ultimate)) {
        return "reference to statement function '"s +
            ultimate.name().ToString() + "'";
      }
      if (scope_.IsDerivedType()) { // C750, C754
        return "reference to function '"s + ultimate.name().ToString() +
            "' not allowed for derived type components or type parameter"
            " values";
      }
      if (auto procChars{characteristics::Procedure::Characterize(
              x.proc(), context_, /*emitError=*/true)}) {
        const auto iter{std::find_if(procChars->dummyArguments.begin(),
            procChars->dummyArguments.end(),
            [](const characteristics::DummyArgument &dummy) {
              return std::holds_alternative<characteristics::DummyProcedure>(
                  dummy.u);
            })};
        if (iter != procChars->dummyArguments.end() &&
            ultimate.name().ToString() != "__builtin_c_funloc") {
          return "reference to function '"s + ultimate.name().ToString() +
              "' with dummy procedure argument '" + iter->name + '\'';
        }
      }
      // References to internal functions are caught in expression semantics.
      // TODO: other checks for standard module procedures
      auto restorer{common::ScopedSet(inInquiry_, false)};
      return (*this)(x.arguments());
    } else { // intrinsic
      const SpecificIntrinsic &intrin{DEREF(x.proc().GetSpecificIntrinsic())};
      bool inInquiry{context_.intrinsics().GetIntrinsicClass(intrin.name) ==
          IntrinsicClass::inquiryFunction};
      if (scope_.IsDerivedType()) { // C750, C754
        if ((context_.intrinsics().IsIntrinsic(intrin.name) &&
                badIntrinsicsForComponents_.find(intrin.name) !=
                    badIntrinsicsForComponents_.end())) {
          return "reference to intrinsic '"s + intrin.name +
              "' not allowed for derived type components or type parameter"
              " values";
        }
        if (inInquiry && !IsConstantExpr(x)) {
          return "non-constant reference to inquiry intrinsic '"s +
              intrin.name +
              "' not allowed for derived type components or type"
              " parameter values";
        }
      }
      // Type-determined inquiries (DIGITS, HUGE, &c.) will have already been
      // folded and won't arrive here.  Inquiries that are represented with
      // DescriptorInquiry operations (LBOUND) are checked elsewhere.  If a
      // call that makes it to here satisfies the requirements of a constant
      // expression (as Fortran defines it), it's fine.
      if (IsConstantExpr(x)) {
        return std::nullopt;
      }
      if (intrin.name == "present") {
        return std::nullopt; // always ok
      }
      const auto &proc{intrin.characteristics.value()};
      std::size_t j{0};
      for (const auto &arg : x.arguments()) {
        bool checkArg{true};
        if (const auto *dataDummy{j < proc.dummyArguments.size()
                    ? std::get_if<characteristics::DummyDataObject>(
                          &proc.dummyArguments[j].u)
                    : nullptr}) {
          if (dataDummy->attrs.test(characteristics::DummyDataObject::Attr::
                      OnlyIntrinsicInquiry)) {
            checkArg = false; // value unused, e.g. IEEE_SUPPORT_FLAG(,,,. X)
          }
        }
        if (arg && checkArg) {
          // Catch CHARACTER(:), ALLOCATABLE :: X; CHARACTER(LEN(X)) :: Y
          if (inInquiry) {
            if (auto dataRef{ExtractDataRef(*arg, true, true)}) {
              if (intrin.name == "allocated" || intrin.name == "associated" ||
                  intrin.name == "is_contiguous") { // ok
              } else if (intrin.name == "len" &&
                  IsPermissibleInquiry(dataRef->GetFirstSymbol(),
                      dataRef->GetLastSymbol(),
                      DescriptorInquiry::Field::Len)) { // ok
              } else if (intrin.name == "lbound" &&
                  IsPermissibleInquiry(dataRef->GetFirstSymbol(),
                      dataRef->GetLastSymbol(),
                      DescriptorInquiry::Field::LowerBound)) { // ok
              } else if ((intrin.name == "shape" || intrin.name == "size" ||
                             intrin.name == "sizeof" ||
                             intrin.name == "storage_size" ||
                             intrin.name == "ubound") &&
                  IsPermissibleInquiry(dataRef->GetFirstSymbol(),
                      dataRef->GetLastSymbol(),
                      DescriptorInquiry::Field::Extent)) { // ok
              } else {
                return "non-constant inquiry function '"s + intrin.name +
                    "' not allowed for local object";
              }
            }
          }
          auto restorer{common::ScopedSet(inInquiry_, inInquiry)};
          if (auto err{(*this)(*arg)}) {
            return err;
          }
        }
        ++j;
      }
      return std::nullopt;
    }
  }

private:
  const semantics::Scope &scope_;
  FoldingContext &context_;
  // Contextual information: this flag is true when in an argument to
  // an inquiry intrinsic like SIZE().
  mutable bool inInquiry_{false};
  bool forElementalFunctionResult_{false}; // F'2023 C15121
  const std::set<std::string> badIntrinsicsForComponents_{
      "allocated", "associated", "extends_type_of", "present", "same_type_as"};

  bool IsInquiryAlwaysPermissible(const semantics::Symbol &) const;
  bool IsPermissibleInquiry(const semantics::Symbol &firstSymbol,
      const semantics::Symbol &lastSymbol,
      DescriptorInquiry::Field field) const;
};

bool CheckSpecificationExprHelper::IsInquiryAlwaysPermissible(
    const semantics::Symbol &symbol) const {
  if (&symbol.owner() != &scope_ || symbol.has<semantics::UseDetails>() ||
      symbol.owner().kind() == semantics::Scope::Kind::Module ||
      semantics::FindCommonBlockContaining(symbol) ||
      symbol.has<semantics::HostAssocDetails>()) {
    return true; // it's nonlocal
  } else if (semantics::IsDummy(symbol) && !forElementalFunctionResult_) {
    return true;
  } else {
    return false;
  }
}

bool CheckSpecificationExprHelper::IsPermissibleInquiry(
    const semantics::Symbol &firstSymbol, const semantics::Symbol &lastSymbol,
    DescriptorInquiry::Field field) const {
  if (IsInquiryAlwaysPermissible(firstSymbol)) {
    return true;
  }
  // Inquiries on local objects may not access a deferred bound or length.
  // (This code used to be a switch, but it proved impossible to write it
  // thus without running afoul of bogus warnings from different C++
  // compilers.)
  if (field == DescriptorInquiry::Field::Rank) {
    return true; // always known
  }
  const auto *object{lastSymbol.detailsIf<semantics::ObjectEntityDetails>()};
  if (field == DescriptorInquiry::Field::LowerBound ||
      field == DescriptorInquiry::Field::Extent ||
      field == DescriptorInquiry::Field::Stride) {
    return object && !object->shape().CanBeDeferredShape();
  }
  if (field == DescriptorInquiry::Field::Len) {
    return object && object->type() &&
        object->type()->category() == semantics::DeclTypeSpec::Character &&
        !object->type()->characterTypeSpec().length().isDeferred();
  }
  return false;
}

template <typename A>
void CheckSpecificationExpr(const A &x, const semantics::Scope &scope,
    FoldingContext &context, bool forElementalFunctionResult) {
  CheckSpecificationExprHelper errors{
      scope, context, forElementalFunctionResult};
  if (auto why{errors(x)}) {
    context.messages().Say("Invalid specification expression%s: %s"_err_en_US,
        forElementalFunctionResult ? " for elemental function result" : "",
        *why);
  }
}

template void CheckSpecificationExpr(const Expr<SomeType> &,
    const semantics::Scope &, FoldingContext &,
    bool forElementalFunctionResult);
template void CheckSpecificationExpr(const Expr<SomeInteger> &,
    const semantics::Scope &, FoldingContext &,
    bool forElementalFunctionResult);
template void CheckSpecificationExpr(const Expr<SubscriptInteger> &,
    const semantics::Scope &, FoldingContext &,
    bool forElementalFunctionResult);
template void CheckSpecificationExpr(const std::optional<Expr<SomeType>> &,
    const semantics::Scope &, FoldingContext &,
    bool forElementalFunctionResult);
template void CheckSpecificationExpr(const std::optional<Expr<SomeInteger>> &,
    const semantics::Scope &, FoldingContext &,
    bool forElementalFunctionResult);
template void CheckSpecificationExpr(
    const std::optional<Expr<SubscriptInteger>> &, const semantics::Scope &,
    FoldingContext &, bool forElementalFunctionResult);

// IsContiguous() -- 9.5.4
class IsContiguousHelper
    : public AnyTraverse<IsContiguousHelper, std::optional<bool>> {
public:
  using Result = std::optional<bool>; // tri-state
  using Base = AnyTraverse<IsContiguousHelper, Result>;
  explicit IsContiguousHelper(FoldingContext &c,
      bool namedConstantSectionsAreContiguous,
      bool firstDimensionStride1 = false)
      : Base{*this}, context_{c},
        namedConstantSectionsAreContiguous_{namedConstantSectionsAreContiguous},
        firstDimensionStride1_{firstDimensionStride1} {}
  using Base::operator();

  template <typename T> Result operator()(const Constant<T> &) const {
    return true;
  }
  Result operator()(const StaticDataObject &) const { return true; }
  Result operator()(const semantics::Symbol &symbol) const {
    const auto &ultimate{symbol.GetUltimate()};
    if (ultimate.attrs().test(semantics::Attr::CONTIGUOUS)) {
      return true;
    } else if (!IsVariable(symbol)) {
      return true;
    } else if (ultimate.Rank() == 0) {
      // Extension: accept scalars as a degenerate case of
      // simple contiguity to allow their use in contexts like
      // data targets in pointer assignments with remapping.
      return true;
    } else if (const auto *details{
                   ultimate.detailsIf<semantics::AssocEntityDetails>()}) {
      // RANK(*) associating entity is contiguous.
      if (details->IsAssumedSize()) {
        return true;
      } else if (!IsVariable(details->expr()) &&
          (namedConstantSectionsAreContiguous_ ||
              !ExtractDataRef(details->expr(), true, true))) {
        // Selector is associated to an expression value.
        return true;
      } else {
        return Base::operator()(ultimate); // use expr
      }
    } else if (semantics::IsPointer(ultimate) || IsAssumedShape(ultimate) ||
        IsAssumedRank(ultimate)) {
      return std::nullopt;
    } else if (ultimate.has<semantics::ObjectEntityDetails>()) {
      return true;
    } else {
      return Base::operator()(ultimate);
    }
  }

  Result operator()(const ArrayRef &x) const {
    if (x.Rank() == 0) {
      return true; // scalars considered contiguous
    }
    int subscriptRank{0};
    auto baseLbounds{GetLBOUNDs(context_, x.base())};
    auto baseUbounds{GetUBOUNDs(context_, x.base())};
    auto subscripts{CheckSubscripts(
        x.subscript(), subscriptRank, &baseLbounds, &baseUbounds)};
    if (!subscripts.value_or(false)) {
      return subscripts; // subscripts not known to be contiguous
    } else if (subscriptRank > 0) {
      // a(1)%b(:,:) is contiguous if and only if a(1)%b is contiguous.
      return (*this)(x.base());
    } else {
      // a(:)%b(1,1) is (probably) not contiguous.
      return std::nullopt;
    }
  }
  Result operator()(const CoarrayRef &x) const { return (*this)(x.base()); }
  Result operator()(const Component &x) const {
    if (x.base().Rank() == 0) {
      return (*this)(x.GetLastSymbol());
    } else {
      const DataRef &base{x.base()};
      if (Result baseIsContiguous{(*this)(base)}) {
        if (!*baseIsContiguous) {
          return false;
        } else {
          bool sizeKnown{false};
          if (auto constShape{GetConstantExtents(context_, x)}) {
            sizeKnown = true;
            if (GetSize(*constShape) <= 1) {
              return true; // empty or singleton
            }
          }
          const Symbol &last{base.GetLastSymbol()};
          if (auto type{DynamicType::From(last)}) {
            CHECK(type->category() == TypeCategory::Derived);
            if (!type->IsPolymorphic()) {
              const auto &derived{type->GetDerivedTypeSpec()};
              if (const auto *scope{derived.scope()}) {
                auto iter{scope->begin()};
                if (++iter == scope->end()) {
                  return true; // type has but one component
                } else if (sizeKnown) {
                  return false; // multiple components & array size is known > 1
                }
              }
            }
          }
        }
      }
      return std::nullopt;
    }
  }
  Result operator()(const ComplexPart &x) const {
    // TODO: should be true when base is empty array or singleton, too
    return x.complex().Rank() == 0;
  }
  Result operator()(const Substring &x) const {
    if (x.Rank() == 0) {
      return true; // scalar substring always contiguous
    }
    // Substrings with rank must have DataRefs as their parents
    const DataRef &parentDataRef{DEREF(x.GetParentIf<DataRef>())};
    std::optional<std::int64_t> len;
    if (auto lenExpr{parentDataRef.LEN()}) {
      len = ToInt64(Fold(context_, std::move(*lenExpr)));
      if (len) {
        if (*len <= 0) {
          return true; // empty substrings
        } else if (*len == 1) {
          // Substrings can't be incomplete; is base array contiguous?
          return (*this)(parentDataRef);
        }
      }
    }
    std::optional<std::int64_t> upper;
    bool upperIsLen{false};
    if (auto upperExpr{x.upper()}) {
      upper = ToInt64(Fold(context_, common::Clone(*upperExpr)));
      if (upper) {
        if (*upper < 1) {
          return true; // substring(n:0) empty
        }
        upperIsLen = len && *upper >= *len;
      } else if (const auto *inquiry{
                     UnwrapConvertedExpr<DescriptorInquiry>(*upperExpr)};
                 inquiry && inquiry->field() == DescriptorInquiry::Field::Len) {
        upperIsLen =
            &parentDataRef.GetLastSymbol() == &inquiry->base().GetLastSymbol();
      }
    } else {
      upperIsLen = true; // substring(n:)
    }
    if (auto lower{ToInt64(Fold(context_, x.lower()))}) {
      if (*lower == 1 && upperIsLen) {
        // known complete substring; is base contiguous?
        return (*this)(parentDataRef);
      } else if (upper) {
        if (*upper < *lower) {
          return true; // empty substring(3:2)
        } else if (*lower > 1) {
          return false; // known incomplete substring
        } else if (len && *upper < *len) {
          return false; // known incomplete substring
        }
      }
    }
    return std::nullopt; // contiguity not known
  }

  Result operator()(const ProcedureRef &x) const {
    if (auto chars{characteristics::Procedure::Characterize(
            x.proc(), context_, /*emitError=*/true)}) {
      if (chars->functionResult) {
        const auto &result{*chars->functionResult};
        if (!result.IsProcedurePointer()) {
          if (result.attrs.test(
                  characteristics::FunctionResult::Attr::Contiguous)) {
            return true;
          }
          if (!result.attrs.test(
                  characteristics::FunctionResult::Attr::Pointer)) {
            return true;
          }
          if (const auto *type{result.GetTypeAndShape()};
              type && type->Rank() == 0) {
            return true; // pointer to scalar
          }
          // Must be non-CONTIGUOUS pointer to array
        }
      }
    }
    return std::nullopt;
  }

  Result operator()(const NullPointer &) const { return true; }

private:
  // Returns "true" for a provably empty or simply contiguous array section;
  // return "false" for a provably nonempty discontiguous section or for use
  // of a vector subscript.
  std::optional<bool> CheckSubscripts(const std::vector<Subscript> &subscript,
      int &rank, const Shape *baseLbounds = nullptr,
      const Shape *baseUbounds = nullptr) const {
    bool anyTriplet{false};
    rank = 0;
    // Detect any provably empty dimension in this array section, which would
    // render the whole section empty and therefore vacuously contiguous.
    std::optional<bool> result;
    bool mayBeEmpty{false};
    auto dims{subscript.size()};
    std::vector<bool> knownPartialSlice(dims, false);
    for (auto j{dims}; j-- > 0;) {
      if (j == 0 && firstDimensionStride1_ && !result.value_or(true)) {
        result.reset(); // ignore problems on later dimensions
      }
      std::optional<ConstantSubscript> dimLbound;
      std::optional<ConstantSubscript> dimUbound;
      std::optional<ConstantSubscript> dimExtent;
      if (baseLbounds && j < baseLbounds->size()) {
        if (const auto &lb{baseLbounds->at(j)}) {
          dimLbound = ToInt64(Fold(context_, Expr<SubscriptInteger>{*lb}));
        }
      }
      if (baseUbounds && j < baseUbounds->size()) {
        if (const auto &ub{baseUbounds->at(j)}) {
          dimUbound = ToInt64(Fold(context_, Expr<SubscriptInteger>{*ub}));
        }
      }
      if (dimLbound && dimUbound) {
        if (*dimLbound <= *dimUbound) {
          dimExtent = *dimUbound - *dimLbound + 1;
        } else {
          // This is an empty dimension.
          result = true;
          dimExtent = 0;
        }
      }
      if (const auto *triplet{std::get_if<Triplet>(&subscript[j].u)}) {
        ++rank;
        const Expr<SubscriptInteger> *lowerBound{triplet->GetLower()};
        const Expr<SubscriptInteger> *upperBound{triplet->GetUpper()};
        std::optional<ConstantSubscript> lowerVal{lowerBound
                ? ToInt64(Fold(context_, Expr<SubscriptInteger>{*lowerBound}))
                : dimLbound};
        std::optional<ConstantSubscript> upperVal{upperBound
                ? ToInt64(Fold(context_, Expr<SubscriptInteger>{*upperBound}))
                : dimUbound};
        if (auto stride{ToInt64(triplet->stride())}) {
          if (j == 0 && *stride == 1 && firstDimensionStride1_) {
            result = *stride == 1; // contiguous or empty if so
          }
          if (lowerVal && upperVal) {
            if (*lowerVal < *upperVal) {
              if (*stride < 0) {
                result = true; // empty dimension
              } else if (!result && *stride > 1 &&
                  *lowerVal + *stride <= *upperVal) {
                result = false; // discontiguous if not empty
              }
            } else if (*lowerVal > *upperVal) {
              if (*stride > 0) {
                result = true; // empty dimension
              } else if (!result && *stride < 0 &&
                  *lowerVal + *stride >= *upperVal) {
                result = false; // discontiguous if not empty
              }
            } else { // bounds known and equal
              if (j == 0 && firstDimensionStride1_) {
                result = true; // stride doesn't matter
              }
            }
          } else { // bounds not both known
            mayBeEmpty = true;
          }
        } else { // stride not known
          if (lowerVal && upperVal && *lowerVal == *upperVal) {
            // stride doesn't matter when bounds are equal
            if (j == 0 && firstDimensionStride1_) {
              result = true;
            }
          } else {
            mayBeEmpty = true;
          }
        }
      } else if (subscript[j].Rank() > 0) { // vector subscript
        ++rank;
        if (!result) {
          result = false;
        }
        mayBeEmpty = true;
      } else { // scalar subscript
        if (dimExtent && *dimExtent > 1) {
          knownPartialSlice[j] = true;
        }
      }
    }
    if (rank == 0) {
      result = true; // scalar
    }
    if (result) {
      return result;
    }
    // Not provably contiguous or discontiguous at this point.
    // Return "true" if simply contiguous, otherwise nullopt.
    for (auto j{subscript.size()}; j-- > 0;) {
      if (const auto *triplet{std::get_if<Triplet>(&subscript[j].u)}) {
        auto stride{ToInt64(triplet->stride())};
        if (!stride || stride != 1) {
          return std::nullopt;
        } else if (anyTriplet) {
          if (triplet->GetLower() || triplet->GetUpper()) {
            // all triplets before the last one must be just ":" for
            // simple contiguity
            return std::nullopt;
          }
        } else {
          anyTriplet = true;
        }
        ++rank;
      } else if (anyTriplet) {
        // If the section cannot be empty, and this dimension's
        // scalar subscript is known not to cover the whole
        // dimension, then the array section is provably
        // discontiguous.
        return (mayBeEmpty || !knownPartialSlice[j])
            ? std::nullopt
            : std::make_optional(false);
      }
    }
    return true; // simply contiguous
  }

  FoldingContext &context_;
  bool namedConstantSectionsAreContiguous_{false};
  bool firstDimensionStride1_{false};
};

template <typename A>
std::optional<bool> IsContiguous(const A &x, FoldingContext &context,
    bool namedConstantSectionsAreContiguous, bool firstDimensionStride1) {
  if (!IsVariable(x) &&
      (namedConstantSectionsAreContiguous || !ExtractDataRef(x, true, true))) {
    return true;
  } else {
    return IsContiguousHelper{
        context, namedConstantSectionsAreContiguous, firstDimensionStride1}(x);
  }
}

std::optional<bool> IsContiguous(const ActualArgument &actual,
    FoldingContext &fc, bool namedConstantSectionsAreContiguous,
    bool firstDimensionStride1) {
  if (auto *expr{actual.UnwrapExpr()}) {
    return IsContiguous(
        *expr, fc, namedConstantSectionsAreContiguous, firstDimensionStride1);
  } else {
    return std::nullopt;
  }
}

template std::optional<bool> IsContiguous(const Expr<SomeType> &,
    FoldingContext &, bool namedConstantSectionsAreContiguous,
    bool firstDimensionStride1);
template std::optional<bool> IsContiguous(const ActualArgument &,
    FoldingContext &, bool namedConstantSectionsAreContiguous,
    bool firstDimensionStride1);
template std::optional<bool> IsContiguous(const ArrayRef &, FoldingContext &,
    bool namedConstantSectionsAreContiguous, bool firstDimensionStride1);
template std::optional<bool> IsContiguous(const Substring &, FoldingContext &,
    bool namedConstantSectionsAreContiguous, bool firstDimensionStride1);
template std::optional<bool> IsContiguous(const Component &, FoldingContext &,
    bool namedConstantSectionsAreContiguous, bool firstDimensionStride1);
template std::optional<bool> IsContiguous(const ComplexPart &, FoldingContext &,
    bool namedConstantSectionsAreContiguous, bool firstDimensionStride1);
template std::optional<bool> IsContiguous(const CoarrayRef &, FoldingContext &,
    bool namedConstantSectionsAreContiguous, bool firstDimensionStride1);
template std::optional<bool> IsContiguous(const Symbol &, FoldingContext &,
    bool namedConstantSectionsAreContiguous, bool firstDimensionStride1);

// IsErrorExpr()
struct IsErrorExprHelper : public AnyTraverse<IsErrorExprHelper, bool> {
  using Result = bool;
  using Base = AnyTraverse<IsErrorExprHelper, Result>;
  IsErrorExprHelper() : Base{*this} {}
  using Base::operator();

  bool operator()(const SpecificIntrinsic &x) {
    return x.name == IntrinsicProcTable::InvalidName;
  }
};

template <typename A> bool IsErrorExpr(const A &x) {
  return IsErrorExprHelper{}(x);
}

template bool IsErrorExpr(const Expr<SomeType> &);

// C1577
// TODO: Also check C1579 & C1582 here
class StmtFunctionChecker
    : public AnyTraverse<StmtFunctionChecker, std::optional<parser::Message>> {
public:
  using Result = std::optional<parser::Message>;
  using Base = AnyTraverse<StmtFunctionChecker, Result>;

  static constexpr auto feature{
      common::LanguageFeature::StatementFunctionExtensions};

  StmtFunctionChecker(const Symbol &sf, FoldingContext &context)
      : Base{*this}, sf_{sf}, context_{context} {
    if (!context_.languageFeatures().IsEnabled(feature)) {
      severity_ = parser::Severity::Error;
    } else if (context_.languageFeatures().ShouldWarn(feature)) {
      severity_ = parser::Severity::Portability;
    }
  }
  using Base::operator();

  Result Return(parser::Message &&msg) const {
    if (severity_) {
      msg.set_severity(*severity_);
      if (*severity_ != parser::Severity::Error) {
        msg.set_languageFeature(feature);
      }
    }
    return std::move(msg);
  }

  template <typename T> Result operator()(const ArrayConstructor<T> &) const {
    if (severity_) {
      return Return(parser::Message{sf_.name(),
          "Statement function '%s' should not contain an array constructor"_port_en_US,
          sf_.name()});
    } else {
      return std::nullopt;
    }
  }
  Result operator()(const StructureConstructor &) const {
    if (severity_) {
      return Return(parser::Message{sf_.name(),
          "Statement function '%s' should not contain a structure constructor"_port_en_US,
          sf_.name()});
    } else {
      return std::nullopt;
    }
  }
  Result operator()(const TypeParamInquiry &) const {
    if (severity_) {
      return Return(parser::Message{sf_.name(),
          "Statement function '%s' should not contain a type parameter inquiry"_port_en_US,
          sf_.name()});
    } else {
      return std::nullopt;
    }
  }
  Result operator()(const ProcedureDesignator &proc) const {
    if (const Symbol * symbol{proc.GetSymbol()}) {
      const Symbol &ultimate{symbol->GetUltimate()};
      if (const auto *subp{
              ultimate.detailsIf<semantics::SubprogramDetails>()}) {
        if (subp->stmtFunction() && &ultimate.owner() == &sf_.owner()) {
          if (ultimate.name().begin() > sf_.name().begin()) {
            return parser::Message{sf_.name(),
                "Statement function '%s' may not reference another statement function '%s' that is defined later"_err_en_US,
                sf_.name(), ultimate.name()};
          }
        }
      }
      if (auto chars{characteristics::Procedure::Characterize(
              proc, context_, /*emitError=*/true)}) {
        if (!chars->CanBeCalledViaImplicitInterface()) {
          if (severity_) {
            return Return(parser::Message{sf_.name(),
                "Statement function '%s' should not reference function '%s' that requires an explicit interface"_port_en_US,
                sf_.name(), symbol->name()});
          }
        }
      }
    }
    if (proc.Rank() > 0) {
      if (severity_) {
        return Return(parser::Message{sf_.name(),
            "Statement function '%s' should not reference a function that returns an array"_port_en_US,
            sf_.name()});
      }
    }
    return std::nullopt;
  }
  Result operator()(const ActualArgument &arg) const {
    if (const auto *expr{arg.UnwrapExpr()}) {
      if (auto result{(*this)(*expr)}) {
        return result;
      }
      if (expr->Rank() > 0 && !UnwrapWholeSymbolOrComponentDataRef(*expr)) {
        if (severity_) {
          return Return(parser::Message{sf_.name(),
              "Statement function '%s' should not pass an array argument that is not a whole array"_port_en_US,
              sf_.name()});
        }
      }
    }
    return std::nullopt;
  }

private:
  const Symbol &sf_;
  FoldingContext &context_;
  std::optional<parser::Severity> severity_;
};

std::optional<parser::Message> CheckStatementFunction(
    const Symbol &sf, const Expr<SomeType> &expr, FoldingContext &context) {
  return StmtFunctionChecker{sf, context}(expr);
}

// Helper class for checking differences between actual and dummy arguments
class CopyInOutExplicitInterface {
public:
  explicit CopyInOutExplicitInterface(FoldingContext &fc,
      const ActualArgument &actual,
      const characteristics::DummyDataObject &dummyObj)
      : fc_{fc}, actual_{actual}, dummyObj_{dummyObj} {}

  // Returns true, if actual and dummy have different contiguity requirements
  bool HaveContiguityDifferences() const {
    // Check actual contiguity, unless dummy doesn't care
    bool dummyTreatAsArray{dummyObj_.ignoreTKR.test(common::IgnoreTKR::Rank)};
    bool actualTreatAsContiguous{
        dummyObj_.ignoreTKR.test(common::IgnoreTKR::Contiguous) ||
        IsSimplyContiguous(actual_, fc_)};
    bool dummyIsExplicitShape{dummyObj_.type.IsExplicitShape()};
    bool dummyIsAssumedSize{dummyObj_.type.attrs().test(
        characteristics::TypeAndShape::Attr::AssumedSize)};
    bool dummyIsPolymorphic{dummyObj_.type.type().IsPolymorphic()};
    // type(*) with IGNORE_TKR(tkr) is often used to interface with C "void*".
    // Since the other languages don't know about Fortran's discontiguity
    // handling, such cases should require contiguity.
    bool dummyIsVoidStar{dummyObj_.type.type().IsAssumedType() &&
        dummyObj_.ignoreTKR.test(common::IgnoreTKR::Type) &&
        dummyObj_.ignoreTKR.test(common::IgnoreTKR::Rank) &&
        dummyObj_.ignoreTKR.test(common::IgnoreTKR::Kind)};
    // Explicit shape and assumed size arrays must be contiguous
    bool dummyNeedsContiguity{dummyIsExplicitShape || dummyIsAssumedSize ||
        (dummyTreatAsArray && !dummyIsPolymorphic) || dummyIsVoidStar ||
        dummyObj_.attrs.test(
            characteristics::DummyDataObject::Attr::Contiguous)};
    return !actualTreatAsContiguous && dummyNeedsContiguity;
  }

  // Returns true, if actual and dummy have polymorphic differences
  bool HavePolymorphicDifferences() const {
    bool dummyIsAssumedRank{dummyObj_.type.attrs().test(
        characteristics::TypeAndShape::Attr::AssumedRank)};
    bool actualIsAssumedRank{semantics::IsAssumedRank(actual_)};
    bool dummyIsAssumedShape{dummyObj_.type.attrs().test(
        characteristics::TypeAndShape::Attr::AssumedShape)};
    bool actualIsAssumedShape{semantics::IsAssumedShape(actual_)};
    if ((actualIsAssumedRank && dummyIsAssumedRank) ||
        (actualIsAssumedShape && dummyIsAssumedShape)) {
      // Assumed-rank and assumed-shape arrays are represented by descriptors,
      // so don't need to do polymorphic check.
    } else if (!dummyObj_.ignoreTKR.test(common::IgnoreTKR::Type)) {
      // flang supports limited cases of passing polymorphic to non-polimorphic.
      // These cases require temporary of non-polymorphic type. (For example,
      // the actual argument could be polymorphic array of child type,
      // while the dummy argument could be non-polymorphic array of parent
      // type.)
      bool dummyIsPolymorphic{dummyObj_.type.type().IsPolymorphic()};
      auto actualType{
          characteristics::TypeAndShape::Characterize(actual_, fc_)};
      bool actualIsPolymorphic{
          actualType && actualType->type().IsPolymorphic()};
      if (actualIsPolymorphic && !dummyIsPolymorphic) {
        return true;
      }
    }
    return false;
  }

  bool HaveArrayOrAssumedRankArgs() const {
    bool dummyTreatAsArray{dummyObj_.ignoreTKR.test(common::IgnoreTKR::Rank)};
    return IsArrayOrAssumedRank(actual_) &&
        (IsArrayOrAssumedRank(dummyObj_) || dummyTreatAsArray);
  }

  bool PassByValue() const {
    return dummyObj_.attrs.test(characteristics::DummyDataObject::Attr::Value);
  }

  bool HaveCoarrayDifferences() const {
    return ExtractCoarrayRef(actual_) && dummyObj_.type.corank() == 0;
  }

  bool HasIntentOut() const { return dummyObj_.intent == common::Intent::Out; }

  bool HasIntentIn() const { return dummyObj_.intent == common::Intent::In; }

  static bool IsArrayOrAssumedRank(const ActualArgument &actual) {
    return semantics::IsAssumedRank(actual) || actual.Rank() > 0;
  }

  static bool IsArrayOrAssumedRank(
      const characteristics::DummyDataObject &dummy) {
    return dummy.type.attrs().test(
               characteristics::TypeAndShape::Attr::AssumedRank) ||
        dummy.type.Rank() > 0;
  }

private:
  FoldingContext &fc_;
  const ActualArgument &actual_;
  const characteristics::DummyDataObject &dummyObj_;
};

// If forCopyOut is false, returns if a particular actual/dummy argument
// combination may need a temporary creation with copy-in operation. If
// forCopyOut is true, returns the same for copy-out operation. For
// procedures with explicit interface, it's expected that "dummy" is not null.
// For procedures with implicit interface dummy may be null.
//
// Note that these copy-in and copy-out checks are done from the caller's
// perspective, meaning that for copy-in the caller need to do the copy
// before calling the callee. Similarly, for copy-out the caller is expected
// to do the copy after the callee returns.
bool MayNeedCopy(const ActualArgument *actual,
    const characteristics::DummyArgument *dummy, FoldingContext &fc,
    bool forCopyOut) {
  if (!actual) {
    return false;
  }
  if (actual->isAlternateReturn()) {
    return false;
  }
  const auto *dummyObj{dummy
          ? std::get_if<characteristics::DummyDataObject>(&dummy->u)
          : nullptr};
  const bool forCopyIn = !forCopyOut;
  if (!evaluate::IsVariable(*actual)) {
    // Actual argument expressions that aren’t variables are copy-in, but
    // not copy-out.
    return forCopyIn;
  }
  if (dummyObj) { // Explict interface
    CopyInOutExplicitInterface check{fc, *actual, *dummyObj};
    if (forCopyOut && check.HasIntentIn()) {
      // INTENT(IN) dummy args never need copy-out
      return false;
    }
    if (forCopyIn && check.HasIntentOut()) {
      // INTENT(OUT) dummy args never need copy-in
      return false;
    }
    if (check.PassByValue()) {
      // Pass by value, always copy-in, never copy-out
      return forCopyIn;
    }
    if (check.HaveCoarrayDifferences()) {
      return true;
    }
    // Note: contiguity and polymorphic checks deal with array or assumed rank
    // arguments
    if (!check.HaveArrayOrAssumedRankArgs()) {
      return false;
    }
    if (check.HaveContiguityDifferences()) {
      return true;
    }
    if (check.HavePolymorphicDifferences()) {
      return true;
    }
  } else { // Implicit interface
    if (ExtractCoarrayRef(*actual)) {
      // Coindexed actual args may need copy-in and copy-out with implicit
      // interface
      return true;
    }
    if (!IsSimplyContiguous(*actual, fc)) {
      // Copy-in:  actual arguments that are variables are copy-in when
      //           non-contiguous.
      // Copy-out: vector subscripts could refer to duplicate elements, can't
      //           copy out.
      return !(forCopyOut && HasVectorSubscript(*actual));
    }
  }
  // For everything else, no copy-in or copy-out
  return false;
}

} // namespace Fortran::evaluate
