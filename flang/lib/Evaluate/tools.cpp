//===-- lib/Evaluate/tools.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/tools.h"
#include "flang/Common/idioms.h"
#include "flang/Evaluate/characteristics.h"
#include "flang/Evaluate/traverse.h"
#include "flang/Parser/message.h"
#include "flang/Semantics/tools.h"
#include <algorithm>
#include <variant>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

// Can x*(a,b) be represented as (x*a,x*b)?  This code duplication
// of the subexpression "x" cannot (yet?) be reliably undone by
// common subexpression elimination in lowering, so it's disabled
// here for now to avoid the risk of potential duplication of
// expensive subexpressions (e.g., large array expressions, references
// to expensive functions) in generate code.
static constexpr bool allowOperandDuplication{false};

std::optional<Expr<SomeType>> AsGenericExpr(DataRef &&ref) {
  if (auto dyType{DynamicType::From(ref.GetLastSymbol())}) {
    return TypedWrapper<Designator, DataRef>(*dyType, std::move(ref));
  } else {
    return std::nullopt;
  }
}

std::optional<Expr<SomeType>> AsGenericExpr(const Symbol &symbol) {
  return AsGenericExpr(DataRef{symbol});
}

Expr<SomeType> Parenthesize(Expr<SomeType> &&expr) {
  return common::visit(
      [&](auto &&x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (common::HasMember<T, TypelessExpression>) {
          return expr; // no parentheses around typeless
        } else if constexpr (std::is_same_v<T, Expr<SomeDerived>>) {
          return AsGenericExpr(Parentheses<SomeDerived>{std::move(x)});
        } else {
          return common::visit(
              [](auto &&y) {
                using T = ResultType<decltype(y)>;
                return AsGenericExpr(Parentheses<T>{std::move(y)});
              },
              std::move(x.u));
        }
      },
      std::move(expr.u));
}

std::optional<DataRef> ExtractDataRef(
    const ActualArgument &arg, bool intoSubstring, bool intoComplexPart) {
  return ExtractDataRef(arg.UnwrapExpr(), intoSubstring, intoComplexPart);
}

std::optional<DataRef> ExtractSubstringBase(const Substring &substring) {
  return common::visit(
      common::visitors{
          [&](const DataRef &x) -> std::optional<DataRef> { return x; },
          [&](const StaticDataObject::Pointer &) -> std::optional<DataRef> {
            return std::nullopt;
          },
      },
      substring.parent());
}

// IsVariable()

auto IsVariableHelper::operator()(const Symbol &symbol) const -> Result {
  // ASSOCIATE(x => expr) -- x counts as a variable, but undefinable
  const Symbol &ultimate{symbol.GetUltimate()};
  return !IsNamedConstant(ultimate) &&
      (ultimate.has<semantics::ObjectEntityDetails>() ||
          (ultimate.has<semantics::EntityDetails>() &&
              ultimate.attrs().test(semantics::Attr::TARGET)) ||
          ultimate.has<semantics::AssocEntityDetails>());
}
auto IsVariableHelper::operator()(const Component &x) const -> Result {
  const Symbol &comp{x.GetLastSymbol()};
  return (*this)(comp) && (IsPointer(comp) || (*this)(x.base()));
}
auto IsVariableHelper::operator()(const ArrayRef &x) const -> Result {
  return (*this)(x.base());
}
auto IsVariableHelper::operator()(const Substring &x) const -> Result {
  return (*this)(x.GetBaseObject());
}
auto IsVariableHelper::operator()(const ProcedureDesignator &x) const
    -> Result {
  if (const Symbol * symbol{x.GetSymbol()}) {
    const Symbol *result{FindFunctionResult(*symbol)};
    return result && IsPointer(*result) && !IsProcedurePointer(*result);
  }
  return false;
}

// Conversions of COMPLEX component expressions to REAL.
ConvertRealOperandsResult ConvertRealOperands(
    parser::ContextualMessages &messages, Expr<SomeType> &&x,
    Expr<SomeType> &&y, int defaultRealKind) {
  return common::visit(
      common::visitors{
          [&](Expr<SomeInteger> &&ix,
              Expr<SomeInteger> &&iy) -> ConvertRealOperandsResult {
            // Can happen in a CMPLX() constructor.  Per F'2018,
            // both integer operands are converted to default REAL.
            return {AsSameKindExprs<TypeCategory::Real>(
                ConvertToKind<TypeCategory::Real>(
                    defaultRealKind, std::move(ix)),
                ConvertToKind<TypeCategory::Real>(
                    defaultRealKind, std::move(iy)))};
          },
          [&](Expr<SomeInteger> &&ix,
              Expr<SomeUnsigned> &&iy) -> ConvertRealOperandsResult {
            return {AsSameKindExprs<TypeCategory::Real>(
                ConvertToKind<TypeCategory::Real>(
                    defaultRealKind, std::move(ix)),
                ConvertToKind<TypeCategory::Real>(
                    defaultRealKind, std::move(iy)))};
          },
          [&](Expr<SomeUnsigned> &&ix,
              Expr<SomeInteger> &&iy) -> ConvertRealOperandsResult {
            return {AsSameKindExprs<TypeCategory::Real>(
                ConvertToKind<TypeCategory::Real>(
                    defaultRealKind, std::move(ix)),
                ConvertToKind<TypeCategory::Real>(
                    defaultRealKind, std::move(iy)))};
          },
          [&](Expr<SomeUnsigned> &&ix,
              Expr<SomeUnsigned> &&iy) -> ConvertRealOperandsResult {
            return {AsSameKindExprs<TypeCategory::Real>(
                ConvertToKind<TypeCategory::Real>(
                    defaultRealKind, std::move(ix)),
                ConvertToKind<TypeCategory::Real>(
                    defaultRealKind, std::move(iy)))};
          },
          [&](Expr<SomeInteger> &&ix,
              Expr<SomeReal> &&ry) -> ConvertRealOperandsResult {
            return {AsSameKindExprs<TypeCategory::Real>(
                ConvertTo(ry, std::move(ix)), std::move(ry))};
          },
          [&](Expr<SomeUnsigned> &&ix,
              Expr<SomeReal> &&ry) -> ConvertRealOperandsResult {
            return {AsSameKindExprs<TypeCategory::Real>(
                ConvertTo(ry, std::move(ix)), std::move(ry))};
          },
          [&](Expr<SomeReal> &&rx,
              Expr<SomeInteger> &&iy) -> ConvertRealOperandsResult {
            return {AsSameKindExprs<TypeCategory::Real>(
                std::move(rx), ConvertTo(rx, std::move(iy)))};
          },
          [&](Expr<SomeReal> &&rx,
              Expr<SomeUnsigned> &&iy) -> ConvertRealOperandsResult {
            return {AsSameKindExprs<TypeCategory::Real>(
                std::move(rx), ConvertTo(rx, std::move(iy)))};
          },
          [&](Expr<SomeReal> &&rx,
              Expr<SomeReal> &&ry) -> ConvertRealOperandsResult {
            return {AsSameKindExprs<TypeCategory::Real>(
                std::move(rx), std::move(ry))};
          },
          [&](Expr<SomeInteger> &&ix,
              BOZLiteralConstant &&by) -> ConvertRealOperandsResult {
            return {AsSameKindExprs<TypeCategory::Real>(
                ConvertToKind<TypeCategory::Real>(
                    defaultRealKind, std::move(ix)),
                ConvertToKind<TypeCategory::Real>(
                    defaultRealKind, std::move(by)))};
          },
          [&](Expr<SomeUnsigned> &&ix,
              BOZLiteralConstant &&by) -> ConvertRealOperandsResult {
            return {AsSameKindExprs<TypeCategory::Real>(
                ConvertToKind<TypeCategory::Real>(
                    defaultRealKind, std::move(ix)),
                ConvertToKind<TypeCategory::Real>(
                    defaultRealKind, std::move(by)))};
          },
          [&](BOZLiteralConstant &&bx,
              Expr<SomeInteger> &&iy) -> ConvertRealOperandsResult {
            return {AsSameKindExprs<TypeCategory::Real>(
                ConvertToKind<TypeCategory::Real>(
                    defaultRealKind, std::move(bx)),
                ConvertToKind<TypeCategory::Real>(
                    defaultRealKind, std::move(iy)))};
          },
          [&](BOZLiteralConstant &&bx,
              Expr<SomeUnsigned> &&iy) -> ConvertRealOperandsResult {
            return {AsSameKindExprs<TypeCategory::Real>(
                ConvertToKind<TypeCategory::Real>(
                    defaultRealKind, std::move(bx)),
                ConvertToKind<TypeCategory::Real>(
                    defaultRealKind, std::move(iy)))};
          },
          [&](Expr<SomeReal> &&rx,
              BOZLiteralConstant &&by) -> ConvertRealOperandsResult {
            return {AsSameKindExprs<TypeCategory::Real>(
                std::move(rx), ConvertTo(rx, std::move(by)))};
          },
          [&](BOZLiteralConstant &&bx,
              Expr<SomeReal> &&ry) -> ConvertRealOperandsResult {
            return {AsSameKindExprs<TypeCategory::Real>(
                ConvertTo(ry, std::move(bx)), std::move(ry))};
          },
          [&](auto &&, auto &&) -> ConvertRealOperandsResult { // C718
            messages.Say(
                "operands must be INTEGER, UNSIGNED, REAL, or BOZ"_err_en_US);
            return std::nullopt;
          },
      },
      std::move(x.u), std::move(y.u));
}

// Helpers for NumericOperation and its subroutines below.
static std::optional<Expr<SomeType>> NoExpr() { return std::nullopt; }

template <TypeCategory CAT>
std::optional<Expr<SomeType>> Package(Expr<SomeKind<CAT>> &&catExpr) {
  return {AsGenericExpr(std::move(catExpr))};
}
template <TypeCategory CAT>
std::optional<Expr<SomeType>> Package(
    std::optional<Expr<SomeKind<CAT>>> &&catExpr) {
  if (catExpr) {
    return {AsGenericExpr(std::move(*catExpr))};
  } else {
    return std::nullopt;
  }
}

// Mixed REAL+INTEGER operations.  REAL**INTEGER is a special case that
// does not require conversion of the exponent expression.
template <template <typename> class OPR>
std::optional<Expr<SomeType>> MixedRealLeft(
    Expr<SomeReal> &&rx, Expr<SomeInteger> &&iy) {
  return Package(common::visit(
      [&](auto &&rxk) -> Expr<SomeReal> {
        using resultType = ResultType<decltype(rxk)>;
        if constexpr (std::is_same_v<OPR<resultType>, Power<resultType>>) {
          return AsCategoryExpr(
              RealToIntPower<resultType>{std::move(rxk), std::move(iy)});
        }
        // G++ 8.1.0 emits bogus warnings about missing return statements if
        // this statement is wrapped in an "else", as it should be.
        return AsCategoryExpr(OPR<resultType>{
            std::move(rxk), ConvertToType<resultType>(std::move(iy))});
      },
      std::move(rx.u)));
}

template <int KIND>
Expr<SomeComplex> MakeComplex(Expr<Type<TypeCategory::Real, KIND>> &&re,
    Expr<Type<TypeCategory::Real, KIND>> &&im) {
  return AsCategoryExpr(ComplexConstructor<KIND>{std::move(re), std::move(im)});
}

std::optional<Expr<SomeComplex>> ConstructComplex(
    parser::ContextualMessages &messages, Expr<SomeType> &&real,
    Expr<SomeType> &&imaginary, int defaultRealKind) {
  if (auto converted{ConvertRealOperands(
          messages, std::move(real), std::move(imaginary), defaultRealKind)}) {
    return {common::visit(
        [](auto &&pair) {
          return MakeComplex(std::move(pair[0]), std::move(pair[1]));
        },
        std::move(*converted))};
  }
  return std::nullopt;
}

std::optional<Expr<SomeComplex>> ConstructComplex(
    parser::ContextualMessages &messages, std::optional<Expr<SomeType>> &&real,
    std::optional<Expr<SomeType>> &&imaginary, int defaultRealKind) {
  if (auto parts{common::AllPresent(std::move(real), std::move(imaginary))}) {
    return ConstructComplex(messages, std::get<0>(std::move(*parts)),
        std::get<1>(std::move(*parts)), defaultRealKind);
  }
  return std::nullopt;
}

// Extracts the real or imaginary part of the result of a COMPLEX
// expression, when that expression is simple enough to be duplicated.
template <bool GET_IMAGINARY> struct ComplexPartExtractor {
  template <typename A> static std::optional<Expr<SomeReal>> Get(const A &) {
    return std::nullopt;
  }

  template <int KIND>
  static std::optional<Expr<SomeReal>> Get(
      const Parentheses<Type<TypeCategory::Complex, KIND>> &kz) {
    if (auto x{Get(kz.left())}) {
      return AsGenericExpr(AsSpecificExpr(
          Parentheses<Type<TypeCategory::Real, KIND>>{std::move(*x)}));
    } else {
      return std::nullopt;
    }
  }

  template <int KIND>
  static std::optional<Expr<SomeReal>> Get(
      const Negate<Type<TypeCategory::Complex, KIND>> &kz) {
    if (auto x{Get(kz.left())}) {
      return AsGenericExpr(AsSpecificExpr(
          Negate<Type<TypeCategory::Real, KIND>>{std::move(*x)}));
    } else {
      return std::nullopt;
    }
  }

  template <int KIND>
  static std::optional<Expr<SomeReal>> Get(
      const Convert<Type<TypeCategory::Complex, KIND>, TypeCategory::Complex>
          &kz) {
    if (auto x{Get(kz.left())}) {
      return AsGenericExpr(AsSpecificExpr(
          Convert<Type<TypeCategory::Real, KIND>, TypeCategory::Real>{
              AsGenericExpr(std::move(*x))}));
    } else {
      return std::nullopt;
    }
  }

  template <int KIND>
  static std::optional<Expr<SomeReal>> Get(const ComplexConstructor<KIND> &kz) {
    return GET_IMAGINARY ? Get(kz.right()) : Get(kz.left());
  }

  template <int KIND>
  static std::optional<Expr<SomeReal>> Get(
      const Constant<Type<TypeCategory::Complex, KIND>> &kz) {
    if (auto cz{kz.GetScalarValue()}) {
      return AsGenericExpr(
          AsSpecificExpr(GET_IMAGINARY ? cz->AIMAG() : cz->REAL()));
    } else {
      return std::nullopt;
    }
  }

  template <int KIND>
  static std::optional<Expr<SomeReal>> Get(
      const Designator<Type<TypeCategory::Complex, KIND>> &kz) {
    if (const auto *symbolRef{std::get_if<SymbolRef>(&kz.u)}) {
      return AsGenericExpr(AsSpecificExpr(
          Designator<Type<TypeCategory::Complex, KIND>>{ComplexPart{
              DataRef{*symbolRef},
              GET_IMAGINARY ? ComplexPart::Part::IM : ComplexPart::Part::RE}}));
    } else {
      return std::nullopt;
    }
  }

  template <int KIND>
  static std::optional<Expr<SomeReal>> Get(
      const Expr<Type<TypeCategory::Complex, KIND>> &kz) {
    return Get(kz.u);
  }

  static std::optional<Expr<SomeReal>> Get(const Expr<SomeComplex> &z) {
    return Get(z.u);
  }
};

// Convert REAL to COMPLEX of the same kind. Preserving the real operand kind
// and then applying complex operand promotion rules allows the result to have
// the highest precision of REAL and COMPLEX operands as required by Fortran
// 2018 10.9.1.3.
Expr<SomeComplex> PromoteRealToComplex(Expr<SomeReal> &&someX) {
  return common::visit(
      [](auto &&x) {
        using RT = ResultType<decltype(x)>;
        return AsCategoryExpr(ComplexConstructor<RT::kind>{
            std::move(x), AsExpr(Constant<RT>{Scalar<RT>{}})});
      },
      std::move(someX.u));
}

// Handle mixed COMPLEX+REAL (or INTEGER) operations in a better way
// than just converting the second operand to COMPLEX and performing the
// corresponding COMPLEX+COMPLEX operation.
template <template <typename> class OPR, TypeCategory RCAT>
std::optional<Expr<SomeType>> MixedComplexLeft(
    parser::ContextualMessages &messages, const Expr<SomeComplex> &zx,
    const Expr<SomeKind<RCAT>> &iry, [[maybe_unused]] int defaultRealKind) {
  if constexpr (RCAT == TypeCategory::Integer &&
      std::is_same_v<OPR<LargestReal>, Power<LargestReal>>) {
    // COMPLEX**INTEGER is a special case that doesn't convert the exponent.
    return Package(common::visit(
        [&](const auto &zxk) {
          using Ty = ResultType<decltype(zxk)>;
          return AsCategoryExpr(AsExpr(
              RealToIntPower<Ty>{common::Clone(zxk), common::Clone(iry)}));
        },
        zx.u));
  }
  std::optional<Expr<SomeReal>> zr{ComplexPartExtractor<false>{}.Get(zx)};
  std::optional<Expr<SomeReal>> zi{ComplexPartExtractor<true>{}.Get(zx)};
  if (!zr || !zi) {
  } else if constexpr (std::is_same_v<OPR<LargestReal>, Add<LargestReal>> ||
      std::is_same_v<OPR<LargestReal>, Subtract<LargestReal>>) {
    // (a,b) + x -> (a+x, b)
    // (a,b) - x -> (a-x, b)
    if (std::optional<Expr<SomeType>> rr{
            NumericOperation<OPR>(messages, AsGenericExpr(std::move(*zr)),
                AsGenericExpr(common::Clone(iry)), defaultRealKind)}) {
      return Package(ConstructComplex(messages, std::move(*rr),
          AsGenericExpr(std::move(*zi)), defaultRealKind));
    }
  } else if constexpr (allowOperandDuplication &&
      (std::is_same_v<OPR<LargestReal>, Multiply<LargestReal>> ||
          std::is_same_v<OPR<LargestReal>, Divide<LargestReal>>)) {
    // (a,b) * x -> (a*x, b*x)
    // (a,b) / x -> (a/x, b/x)
    auto copy{iry};
    auto rr{NumericOperation<OPR>(messages, AsGenericExpr(std::move(*zr)),
        AsGenericExpr(common::Clone(iry)), defaultRealKind)};
    auto ri{NumericOperation<OPR>(messages, AsGenericExpr(std::move(*zi)),
        AsGenericExpr(std::move(copy)), defaultRealKind)};
    if (auto parts{common::AllPresent(std::move(rr), std::move(ri))}) {
      return Package(ConstructComplex(messages, std::get<0>(std::move(*parts)),
          std::get<1>(std::move(*parts)), defaultRealKind));
    }
  }
  return std::nullopt;
}

// Mixed COMPLEX operations with the COMPLEX operand on the right.
//  x + (a,b) -> (x+a, b)
//  x - (a,b) -> (x-a, -b)
//  x * (a,b) -> (x*a, x*b)
//  x / (a,b) -> (x,0) / (a,b)   (and **)
template <template <typename> class OPR, TypeCategory LCAT>
std::optional<Expr<SomeType>> MixedComplexRight(
    parser::ContextualMessages &messages, const Expr<SomeKind<LCAT>> &irx,
    const Expr<SomeComplex> &zy, [[maybe_unused]] int defaultRealKind) {
  if constexpr (std::is_same_v<OPR<LargestReal>, Add<LargestReal>>) {
    // x + (a,b) -> (a,b) + x -> (a+x, b)
    return MixedComplexLeft<OPR, LCAT>(messages, zy, irx, defaultRealKind);
  } else if constexpr (allowOperandDuplication &&
      std::is_same_v<OPR<LargestReal>, Multiply<LargestReal>>) {
    // x * (a,b) -> (a,b) * x -> (a*x, b*x)
    return MixedComplexLeft<OPR, LCAT>(messages, zy, irx, defaultRealKind);
  } else if constexpr (std::is_same_v<OPR<LargestReal>,
                           Subtract<LargestReal>>) {
    // x - (a,b) -> (x-a, -b)
    std::optional<Expr<SomeReal>> zr{ComplexPartExtractor<false>{}.Get(zy)};
    std::optional<Expr<SomeReal>> zi{ComplexPartExtractor<true>{}.Get(zy)};
    if (zr && zi) {
      if (std::optional<Expr<SomeType>> rr{NumericOperation<Subtract>(messages,
              AsGenericExpr(common::Clone(irx)), AsGenericExpr(std::move(*zr)),
              defaultRealKind)}) {
        return Package(ConstructComplex(messages, std::move(*rr),
            AsGenericExpr(-std::move(*zi)), defaultRealKind));
      }
    }
  }
  return std::nullopt;
}

// Promotes REAL(rk) and COMPLEX(zk) operands COMPLEX(max(rk,zk))
// then combine them with an operator.
template <template <typename> class OPR, TypeCategory XCAT, TypeCategory YCAT>
Expr<SomeComplex> PromoteMixedComplexReal(
    Expr<SomeKind<XCAT>> &&x, Expr<SomeKind<YCAT>> &&y) {
  static_assert(XCAT == TypeCategory::Complex || YCAT == TypeCategory::Complex);
  static_assert(XCAT == TypeCategory::Real || YCAT == TypeCategory::Real);
  return common::visit(
      [&](const auto &kx, const auto &ky) {
        constexpr int maxKind{std::max(
            ResultType<decltype(kx)>::kind, ResultType<decltype(ky)>::kind)};
        using ZTy = Type<TypeCategory::Complex, maxKind>;
        return Expr<SomeComplex>{
            Expr<ZTy>{OPR<ZTy>{ConvertToType<ZTy>(std::move(x)),
                ConvertToType<ZTy>(std::move(y))}}};
      },
      x.u, y.u);
}

// N.B. When a "typeless" BOZ literal constant appears as one (not both!) of
// the operands to a dyadic operation where one is permitted, it assumes the
// type and kind of the other operand.
template <template <typename> class OPR, bool CAN_BE_UNSIGNED>
std::optional<Expr<SomeType>> NumericOperation(
    parser::ContextualMessages &messages, Expr<SomeType> &&x,
    Expr<SomeType> &&y, int defaultRealKind) {
  return common::visit(
      common::visitors{
          [](Expr<SomeInteger> &&ix, Expr<SomeInteger> &&iy) {
            return Package(PromoteAndCombine<OPR, TypeCategory::Integer>(
                std::move(ix), std::move(iy)));
          },
          [](Expr<SomeReal> &&rx, Expr<SomeReal> &&ry) {
            return Package(PromoteAndCombine<OPR, TypeCategory::Real>(
                std::move(rx), std::move(ry)));
          },
          [&](Expr<SomeUnsigned> &&ix, Expr<SomeUnsigned> &&iy) {
            if constexpr (CAN_BE_UNSIGNED) {
              return Package(PromoteAndCombine<OPR, TypeCategory::Unsigned>(
                  std::move(ix), std::move(iy)));
            } else {
              messages.Say("Operands must not be UNSIGNED"_err_en_US);
              return NoExpr();
            }
          },
          // Mixed REAL/INTEGER operations
          [](Expr<SomeReal> &&rx, Expr<SomeInteger> &&iy) {
            return MixedRealLeft<OPR>(std::move(rx), std::move(iy));
          },
          [](Expr<SomeInteger> &&ix, Expr<SomeReal> &&ry) {
            return Package(common::visit(
                [&](auto &&ryk) -> Expr<SomeReal> {
                  using resultType = ResultType<decltype(ryk)>;
                  return AsCategoryExpr(
                      OPR<resultType>{ConvertToType<resultType>(std::move(ix)),
                          std::move(ryk)});
                },
                std::move(ry.u)));
          },
          // Homogeneous and mixed COMPLEX operations
          [](Expr<SomeComplex> &&zx, Expr<SomeComplex> &&zy) {
            return Package(PromoteAndCombine<OPR, TypeCategory::Complex>(
                std::move(zx), std::move(zy)));
          },
          [&](Expr<SomeComplex> &&zx, Expr<SomeInteger> &&iy) {
            if (auto result{
                    MixedComplexLeft<OPR>(messages, zx, iy, defaultRealKind)}) {
              return result;
            } else {
              return Package(PromoteAndCombine<OPR, TypeCategory::Complex>(
                  std::move(zx), ConvertTo(zx, std::move(iy))));
            }
          },
          [&](Expr<SomeComplex> &&zx, Expr<SomeReal> &&ry) {
            if (auto result{
                    MixedComplexLeft<OPR>(messages, zx, ry, defaultRealKind)}) {
              return result;
            } else {
              return Package(
                  PromoteMixedComplexReal<OPR>(std::move(zx), std::move(ry)));
            }
          },
          [&](Expr<SomeInteger> &&ix, Expr<SomeComplex> &&zy) {
            if (auto result{MixedComplexRight<OPR>(
                    messages, ix, zy, defaultRealKind)}) {
              return result;
            } else {
              return Package(PromoteAndCombine<OPR, TypeCategory::Complex>(
                  ConvertTo(zy, std::move(ix)), std::move(zy)));
            }
          },
          [&](Expr<SomeReal> &&rx, Expr<SomeComplex> &&zy) {
            if (auto result{MixedComplexRight<OPR>(
                    messages, rx, zy, defaultRealKind)}) {
              return result;
            } else {
              return Package(
                  PromoteMixedComplexReal<OPR>(std::move(rx), std::move(zy)));
            }
          },
          // Operations with one typeless operand
          [&](BOZLiteralConstant &&bx, Expr<SomeInteger> &&iy) {
            return NumericOperation<OPR, CAN_BE_UNSIGNED>(messages,
                AsGenericExpr(ConvertTo(iy, std::move(bx))), std::move(y),
                defaultRealKind);
          },
          [&](BOZLiteralConstant &&bx, Expr<SomeUnsigned> &&iy) {
            return NumericOperation<OPR, CAN_BE_UNSIGNED>(messages,
                AsGenericExpr(ConvertTo(iy, std::move(bx))), std::move(y),
                defaultRealKind);
          },
          [&](BOZLiteralConstant &&bx, Expr<SomeReal> &&ry) {
            return NumericOperation<OPR, CAN_BE_UNSIGNED>(messages,
                AsGenericExpr(ConvertTo(ry, std::move(bx))), std::move(y),
                defaultRealKind);
          },
          [&](Expr<SomeInteger> &&ix, BOZLiteralConstant &&by) {
            return NumericOperation<OPR, CAN_BE_UNSIGNED>(messages,
                std::move(x), AsGenericExpr(ConvertTo(ix, std::move(by))),
                defaultRealKind);
          },
          [&](Expr<SomeUnsigned> &&ix, BOZLiteralConstant &&by) {
            return NumericOperation<OPR, CAN_BE_UNSIGNED>(messages,
                std::move(x), AsGenericExpr(ConvertTo(ix, std::move(by))),
                defaultRealKind);
          },
          [&](Expr<SomeReal> &&rx, BOZLiteralConstant &&by) {
            return NumericOperation<OPR, CAN_BE_UNSIGNED>(messages,
                std::move(x), AsGenericExpr(ConvertTo(rx, std::move(by))),
                defaultRealKind);
          },
          // Error cases
          [&](Expr<SomeUnsigned> &&, auto &&) {
            messages.Say("Both operands must be UNSIGNED"_err_en_US);
            return NoExpr();
          },
          [&](auto &&, Expr<SomeUnsigned> &&) {
            messages.Say("Both operands must be UNSIGNED"_err_en_US);
            return NoExpr();
          },
          [&](auto &&, auto &&) {
            messages.Say("non-numeric operands to numeric operation"_err_en_US);
            return NoExpr();
          },
      },
      std::move(x.u), std::move(y.u));
}

template std::optional<Expr<SomeType>> NumericOperation<Power, false>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&,
    int defaultRealKind);
template std::optional<Expr<SomeType>> NumericOperation<Multiply>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&,
    int defaultRealKind);
template std::optional<Expr<SomeType>> NumericOperation<Divide>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&,
    int defaultRealKind);
template std::optional<Expr<SomeType>> NumericOperation<Add>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&,
    int defaultRealKind);
template std::optional<Expr<SomeType>> NumericOperation<Subtract>(
    parser::ContextualMessages &, Expr<SomeType> &&, Expr<SomeType> &&,
    int defaultRealKind);

std::optional<Expr<SomeType>> Negation(
    parser::ContextualMessages &messages, Expr<SomeType> &&x) {
  return common::visit(
      common::visitors{
          [&](BOZLiteralConstant &&) {
            messages.Say("BOZ literal cannot be negated"_err_en_US);
            return NoExpr();
          },
          [&](NullPointer &&) {
            messages.Say("NULL() cannot be negated"_err_en_US);
            return NoExpr();
          },
          [&](ProcedureDesignator &&) {
            messages.Say("Subroutine cannot be negated"_err_en_US);
            return NoExpr();
          },
          [&](ProcedureRef &&) {
            messages.Say("Pointer to subroutine cannot be negated"_err_en_US);
            return NoExpr();
          },
          [&](Expr<SomeInteger> &&x) { return Package(-std::move(x)); },
          [&](Expr<SomeReal> &&x) { return Package(-std::move(x)); },
          [&](Expr<SomeComplex> &&x) { return Package(-std::move(x)); },
          [&](Expr<SomeCharacter> &&) {
            messages.Say("CHARACTER cannot be negated"_err_en_US);
            return NoExpr();
          },
          [&](Expr<SomeLogical> &&) {
            messages.Say("LOGICAL cannot be negated"_err_en_US);
            return NoExpr();
          },
          [&](Expr<SomeUnsigned> &&x) { return Package(-std::move(x)); },
          [&](Expr<SomeDerived> &&) {
            messages.Say("Operand cannot be negated"_err_en_US);
            return NoExpr();
          },
      },
      std::move(x.u));
}

Expr<SomeLogical> LogicalNegation(Expr<SomeLogical> &&x) {
  return common::visit(
      [](auto &&xk) { return AsCategoryExpr(LogicalNegation(std::move(xk))); },
      std::move(x.u));
}

template <TypeCategory CAT>
Expr<LogicalResult> PromoteAndRelate(
    RelationalOperator opr, Expr<SomeKind<CAT>> &&x, Expr<SomeKind<CAT>> &&y) {
  return common::visit(
      [=](auto &&xy) {
        return PackageRelation(opr, std::move(xy[0]), std::move(xy[1]));
      },
      AsSameKindExprs(std::move(x), std::move(y)));
}

std::optional<Expr<LogicalResult>> Relate(parser::ContextualMessages &messages,
    RelationalOperator opr, Expr<SomeType> &&x, Expr<SomeType> &&y) {
  return common::visit(
      common::visitors{
          [=](Expr<SomeInteger> &&ix,
              Expr<SomeInteger> &&iy) -> std::optional<Expr<LogicalResult>> {
            return PromoteAndRelate(opr, std::move(ix), std::move(iy));
          },
          [=](Expr<SomeUnsigned> &&ix,
              Expr<SomeUnsigned> &&iy) -> std::optional<Expr<LogicalResult>> {
            return PromoteAndRelate(opr, std::move(ix), std::move(iy));
          },
          [=](Expr<SomeReal> &&rx,
              Expr<SomeReal> &&ry) -> std::optional<Expr<LogicalResult>> {
            return PromoteAndRelate(opr, std::move(rx), std::move(ry));
          },
          [&](Expr<SomeReal> &&rx, Expr<SomeInteger> &&iy) {
            return Relate(messages, opr, std::move(x),
                AsGenericExpr(ConvertTo(rx, std::move(iy))));
          },
          [&](Expr<SomeInteger> &&ix, Expr<SomeReal> &&ry) {
            return Relate(messages, opr,
                AsGenericExpr(ConvertTo(ry, std::move(ix))), std::move(y));
          },
          [&](Expr<SomeComplex> &&zx,
              Expr<SomeComplex> &&zy) -> std::optional<Expr<LogicalResult>> {
            if (opr == RelationalOperator::EQ ||
                opr == RelationalOperator::NE) {
              return PromoteAndRelate(opr, std::move(zx), std::move(zy));
            } else {
              messages.Say(
                  "COMPLEX data may be compared only for equality"_err_en_US);
              return std::nullopt;
            }
          },
          [&](Expr<SomeComplex> &&zx, Expr<SomeInteger> &&iy) {
            return Relate(messages, opr, std::move(x),
                AsGenericExpr(ConvertTo(zx, std::move(iy))));
          },
          [&](Expr<SomeComplex> &&zx, Expr<SomeReal> &&ry) {
            return Relate(messages, opr, std::move(x),
                AsGenericExpr(ConvertTo(zx, std::move(ry))));
          },
          [&](Expr<SomeInteger> &&ix, Expr<SomeComplex> &&zy) {
            return Relate(messages, opr,
                AsGenericExpr(ConvertTo(zy, std::move(ix))), std::move(y));
          },
          [&](Expr<SomeReal> &&rx, Expr<SomeComplex> &&zy) {
            return Relate(messages, opr,
                AsGenericExpr(ConvertTo(zy, std::move(rx))), std::move(y));
          },
          [&](Expr<SomeCharacter> &&cx, Expr<SomeCharacter> &&cy) {
            return common::visit(
                [&](auto &&cxk,
                    auto &&cyk) -> std::optional<Expr<LogicalResult>> {
                  using Ty = ResultType<decltype(cxk)>;
                  if constexpr (std::is_same_v<Ty, ResultType<decltype(cyk)>>) {
                    return PackageRelation(opr, std::move(cxk), std::move(cyk));
                  } else {
                    messages.Say(
                        "CHARACTER operands do not have same KIND"_err_en_US);
                    return std::nullopt;
                  }
                },
                std::move(cx.u), std::move(cy.u));
          },
          // Default case
          [&](auto &&, auto &&) {
            DIE("invalid types for relational operator");
            return std::optional<Expr<LogicalResult>>{};
          },
      },
      std::move(x.u), std::move(y.u));
}

Expr<SomeLogical> BinaryLogicalOperation(
    LogicalOperator opr, Expr<SomeLogical> &&x, Expr<SomeLogical> &&y) {
  CHECK(opr != LogicalOperator::Not);
  return common::visit(
      [=](auto &&xy) {
        using Ty = ResultType<decltype(xy[0])>;
        return Expr<SomeLogical>{BinaryLogicalOperation<Ty::kind>(
            opr, std::move(xy[0]), std::move(xy[1]))};
      },
      AsSameKindExprs(std::move(x), std::move(y)));
}

template <TypeCategory TO>
std::optional<Expr<SomeType>> ConvertToNumeric(int kind, Expr<SomeType> &&x) {
  static_assert(common::IsNumericTypeCategory(TO));
  return common::visit(
      [=](auto &&cx) -> std::optional<Expr<SomeType>> {
        using cxType = std::decay_t<decltype(cx)>;
        if constexpr (!common::HasMember<cxType, TypelessExpression>) {
          if constexpr (IsNumericTypeCategory(ResultType<cxType>::category)) {
            return Expr<SomeType>{ConvertToKind<TO>(kind, std::move(cx))};
          }
        }
        return std::nullopt;
      },
      std::move(x.u));
}

std::optional<Expr<SomeType>> ConvertToType(
    const DynamicType &type, Expr<SomeType> &&x) {
  if (type.IsTypelessIntrinsicArgument()) {
    return std::nullopt;
  }
  switch (type.category()) {
  case TypeCategory::Integer:
    if (auto *boz{std::get_if<BOZLiteralConstant>(&x.u)}) {
      // Extension to C7109: allow BOZ literals to appear in integer contexts
      // when the type is unambiguous.
      return Expr<SomeType>{
          ConvertToKind<TypeCategory::Integer>(type.kind(), std::move(*boz))};
    }
    return ConvertToNumeric<TypeCategory::Integer>(type.kind(), std::move(x));
  case TypeCategory::Unsigned:
    if (auto *boz{std::get_if<BOZLiteralConstant>(&x.u)}) {
      return Expr<SomeType>{
          ConvertToKind<TypeCategory::Unsigned>(type.kind(), std::move(*boz))};
    }
    if (auto *cx{UnwrapExpr<Expr<SomeUnsigned>>(x)}) {
      return Expr<SomeType>{
          ConvertToKind<TypeCategory::Unsigned>(type.kind(), std::move(*cx))};
    }
    break;
  case TypeCategory::Real:
    if (auto *boz{std::get_if<BOZLiteralConstant>(&x.u)}) {
      return Expr<SomeType>{
          ConvertToKind<TypeCategory::Real>(type.kind(), std::move(*boz))};
    }
    return ConvertToNumeric<TypeCategory::Real>(type.kind(), std::move(x));
  case TypeCategory::Complex:
    return ConvertToNumeric<TypeCategory::Complex>(type.kind(), std::move(x));
  case TypeCategory::Character:
    if (auto *cx{UnwrapExpr<Expr<SomeCharacter>>(x)}) {
      auto converted{
          ConvertToKind<TypeCategory::Character>(type.kind(), std::move(*cx))};
      if (auto length{type.GetCharLength()}) {
        converted = common::visit(
            [&](auto &&x) {
              using CharacterType = ResultType<decltype(x)>;
              return Expr<SomeCharacter>{
                  Expr<CharacterType>{SetLength<CharacterType::kind>{
                      std::move(x), std::move(*length)}}};
            },
            std::move(converted.u));
      }
      return Expr<SomeType>{std::move(converted)};
    }
    break;
  case TypeCategory::Logical:
    if (auto *cx{UnwrapExpr<Expr<SomeLogical>>(x)}) {
      return Expr<SomeType>{
          ConvertToKind<TypeCategory::Logical>(type.kind(), std::move(*cx))};
    }
    break;
  case TypeCategory::Derived:
    if (auto fromType{x.GetType()}) {
      if (type.IsTkCompatibleWith(*fromType)) {
        // "x" could be assigned or passed to "type", or appear in a
        // structure constructor as a value for a component with "type"
        return std::move(x);
      }
    }
    break;
  }
  return std::nullopt;
}

std::optional<Expr<SomeType>> ConvertToType(
    const DynamicType &to, std::optional<Expr<SomeType>> &&x) {
  if (x) {
    return ConvertToType(to, std::move(*x));
  } else {
    return std::nullopt;
  }
}

std::optional<Expr<SomeType>> ConvertToType(
    const Symbol &symbol, Expr<SomeType> &&x) {
  if (auto symType{DynamicType::From(symbol)}) {
    return ConvertToType(*symType, std::move(x));
  }
  return std::nullopt;
}

std::optional<Expr<SomeType>> ConvertToType(
    const Symbol &to, std::optional<Expr<SomeType>> &&x) {
  if (x) {
    return ConvertToType(to, std::move(*x));
  } else {
    return std::nullopt;
  }
}

bool IsAssumedRank(const Symbol &original) {
  if (const auto *assoc{original.detailsIf<semantics::AssocEntityDetails>()}) {
    if (assoc->rank()) {
      return false; // in RANK(n) or RANK(*)
    } else if (assoc->IsAssumedRank()) {
      return true; // RANK DEFAULT
    }
  }
  const Symbol &symbol{semantics::ResolveAssociations(original)};
  const auto *object{symbol.detailsIf<semantics::ObjectEntityDetails>()};
  return object && object->IsAssumedRank();
}

bool IsAssumedRank(const ActualArgument &arg) {
  if (const auto *expr{arg.UnwrapExpr()}) {
    return IsAssumedRank(*expr);
  } else {
    const Symbol *assumedTypeDummy{arg.GetAssumedTypeDummy()};
    CHECK(assumedTypeDummy);
    return IsAssumedRank(*assumedTypeDummy);
  }
}

int GetCorank(const ActualArgument &arg) {
  const auto *expr{arg.UnwrapExpr()};
  return GetCorank(*expr);
}

bool IsProcedureDesignator(const Expr<SomeType> &expr) {
  return std::holds_alternative<ProcedureDesignator>(expr.u);
}
bool IsFunctionDesignator(const Expr<SomeType> &expr) {
  const auto *designator{std::get_if<ProcedureDesignator>(&expr.u)};
  return designator && designator->GetType().has_value();
}

bool IsPointer(const Expr<SomeType> &expr) {
  return IsObjectPointer(expr) || IsProcedurePointer(expr);
}

bool IsProcedurePointer(const Expr<SomeType> &expr) {
  if (IsNullProcedurePointer(expr)) {
    return true;
  } else if (const auto *funcRef{UnwrapProcedureRef(expr)}) {
    if (const Symbol * proc{funcRef->proc().GetSymbol()}) {
      const Symbol *result{FindFunctionResult(*proc)};
      return result && IsProcedurePointer(*result);
    } else {
      return false;
    }
  } else if (const auto *proc{std::get_if<ProcedureDesignator>(&expr.u)}) {
    return IsProcedurePointer(proc->GetSymbol());
  } else {
    return false;
  }
}

bool IsProcedure(const Expr<SomeType> &expr) {
  return IsProcedureDesignator(expr) || IsProcedurePointer(expr);
}

bool IsProcedurePointerTarget(const Expr<SomeType> &expr) {
  return common::visit(common::visitors{
                           [](const NullPointer &) { return true; },
                           [](const ProcedureDesignator &) { return true; },
                           [](const ProcedureRef &) { return true; },
                           [&](const auto &) {
                             const Symbol *last{GetLastSymbol(expr)};
                             return last && IsProcedurePointer(*last);
                           },
                       },
      expr.u);
}

bool IsObjectPointer(const Expr<SomeType> &expr) {
  if (IsNullObjectPointer(expr)) {
    return true;
  } else if (IsProcedurePointerTarget(expr)) {
    return false;
  } else if (const auto *funcRef{UnwrapProcedureRef(expr)}) {
    return IsVariable(*funcRef);
  } else if (const Symbol * symbol{UnwrapWholeSymbolOrComponentDataRef(expr)}) {
    return IsPointer(symbol->GetUltimate());
  } else {
    return false;
  }
}

// IsNullPointer() & variations

template <bool IS_PROC_PTR> struct IsNullPointerHelper {
  template <typename A> bool operator()(const A &) const { return false; }
  bool operator()(const ProcedureRef &call) const {
    if constexpr (IS_PROC_PTR) {
      const auto *intrinsic{call.proc().GetSpecificIntrinsic()};
      return intrinsic &&
          intrinsic->characteristics.value().attrs.test(
              characteristics::Procedure::Attr::NullPointer);
    } else {
      return false;
    }
  }
  template <typename T> bool operator()(const FunctionRef<T> &call) const {
    if constexpr (IS_PROC_PTR) {
      return false;
    } else {
      const auto *intrinsic{call.proc().GetSpecificIntrinsic()};
      return intrinsic &&
          intrinsic->characteristics.value().attrs.test(
              characteristics::Procedure::Attr::NullPointer);
    }
  }
  template <typename T> bool operator()(const Designator<T> &x) const {
    if (const auto *component{std::get_if<Component>(&x.u)}) {
      if (const auto *baseSym{std::get_if<SymbolRef>(&component->base().u)}) {
        const Symbol &base{**baseSym};
        if (const auto *object{
                base.detailsIf<semantics::ObjectEntityDetails>()}) {
          // TODO: nested component and array references
          if (IsNamedConstant(base) && object->init()) {
            if (auto structCons{
                    GetScalarConstantValue<SomeDerived>(*object->init())}) {
              auto iter{structCons->values().find(component->GetLastSymbol())};
              if (iter != structCons->values().end()) {
                return (*this)(iter->second.value());
              }
            }
          }
        }
      }
    }
    return false;
  }
  bool operator()(const NullPointer &) const { return true; }
  template <typename T> bool operator()(const Parentheses<T> &x) const {
    return (*this)(x.left());
  }
  template <typename T> bool operator()(const Expr<T> &x) const {
    return common::visit(*this, x.u);
  }
};

bool IsNullObjectPointer(const Expr<SomeType> &expr) {
  return IsNullPointerHelper<false>{}(expr);
}

bool IsNullProcedurePointer(const Expr<SomeType> &expr) {
  return IsNullPointerHelper<true>{}(expr);
}

bool IsNullPointer(const Expr<SomeType> &expr) {
  return IsNullObjectPointer(expr) || IsNullProcedurePointer(expr);
}

bool IsBareNullPointer(const Expr<SomeType> *expr) {
  return expr && std::holds_alternative<NullPointer>(expr->u);
}

// GetSymbolVector()
auto GetSymbolVectorHelper::operator()(const Symbol &x) const -> Result {
  if (const auto *details{x.detailsIf<semantics::AssocEntityDetails>()}) {
    if (IsVariable(details->expr()) && !UnwrapProcedureRef(*details->expr())) {
      // associate(x => variable that is not a pointer returned by a function)
      return (*this)(details->expr());
    }
  }
  return {x.GetUltimate()};
}
auto GetSymbolVectorHelper::operator()(const Component &x) const -> Result {
  Result result{(*this)(x.base())};
  result.emplace_back(x.GetLastSymbol());
  return result;
}
auto GetSymbolVectorHelper::operator()(const ArrayRef &x) const -> Result {
  return GetSymbolVector(x.base());
}
auto GetSymbolVectorHelper::operator()(const CoarrayRef &x) const -> Result {
  return x.base();
}

const Symbol *GetLastTarget(const SymbolVector &symbols) {
  auto end{std::crend(symbols)};
  // N.B. Neither clang nor g++ recognizes "symbols.crbegin()" here.
  auto iter{std::find_if(std::crbegin(symbols), end, [](const Symbol &x) {
    return x.attrs().HasAny(
        {semantics::Attr::POINTER, semantics::Attr::TARGET});
  })};
  return iter == end ? nullptr : &**iter;
}

struct CollectSymbolsHelper
    : public SetTraverse<CollectSymbolsHelper, semantics::UnorderedSymbolSet> {
  using Base = SetTraverse<CollectSymbolsHelper, semantics::UnorderedSymbolSet>;
  CollectSymbolsHelper() : Base{*this} {}
  using Base::operator();
  semantics::UnorderedSymbolSet operator()(const Symbol &symbol) const {
    return {symbol};
  }
};
template <typename A> semantics::UnorderedSymbolSet CollectSymbols(const A &x) {
  return CollectSymbolsHelper{}(x);
}
template semantics::UnorderedSymbolSet CollectSymbols(const Expr<SomeType> &);
template semantics::UnorderedSymbolSet CollectSymbols(
    const Expr<SomeInteger> &);
template semantics::UnorderedSymbolSet CollectSymbols(
    const Expr<SubscriptInteger> &);

struct CollectCudaSymbolsHelper : public SetTraverse<CollectCudaSymbolsHelper,
                                      semantics::UnorderedSymbolSet> {
  using Base =
      SetTraverse<CollectCudaSymbolsHelper, semantics::UnorderedSymbolSet>;
  CollectCudaSymbolsHelper() : Base{*this} {}
  using Base::operator();
  semantics::UnorderedSymbolSet operator()(const Symbol &symbol) const {
    return {symbol};
  }
  // Overload some of the operator() to filter out the symbols that are not
  // of interest for CUDA data transfer logic.
  semantics::UnorderedSymbolSet operator()(const DescriptorInquiry &) const {
    return {};
  }
  semantics::UnorderedSymbolSet operator()(const Subscript &) const {
    return {};
  }
  semantics::UnorderedSymbolSet operator()(const ProcedureRef &) const {
    return {};
  }
};
template <typename A>
semantics::UnorderedSymbolSet CollectCudaSymbols(const A &x) {
  return CollectCudaSymbolsHelper{}(x);
}
template semantics::UnorderedSymbolSet CollectCudaSymbols(
    const Expr<SomeType> &);
template semantics::UnorderedSymbolSet CollectCudaSymbols(
    const Expr<SomeInteger> &);
template semantics::UnorderedSymbolSet CollectCudaSymbols(
    const Expr<SubscriptInteger> &);

// HasVectorSubscript()
struct HasVectorSubscriptHelper
    : public AnyTraverse<HasVectorSubscriptHelper, bool,
          /*TraverseAssocEntityDetails=*/false> {
  using Base = AnyTraverse<HasVectorSubscriptHelper, bool, false>;
  HasVectorSubscriptHelper() : Base{*this} {}
  using Base::operator();
  bool operator()(const Subscript &ss) const {
    return !std::holds_alternative<Triplet>(ss.u) && ss.Rank() > 0;
  }
  bool operator()(const ProcedureRef &) const {
    return false; // don't descend into function call arguments
  }
};

bool HasVectorSubscript(const Expr<SomeType> &expr) {
  return HasVectorSubscriptHelper{}(expr);
}

// HasConstant()
struct HasConstantHelper : public AnyTraverse<HasConstantHelper, bool,
                               /*TraverseAssocEntityDetails=*/false> {
  using Base = AnyTraverse<HasConstantHelper, bool, false>;
  HasConstantHelper() : Base{*this} {}
  using Base::operator();
  template <typename T> bool operator()(const Constant<T> &) const {
    return true;
  }
  // Only look for constant not in subscript.
  bool operator()(const Subscript &) const { return false; }
};

bool HasConstant(const Expr<SomeType> &expr) {
  return HasConstantHelper{}(expr);
}

parser::Message *AttachDeclaration(
    parser::Message &message, const Symbol &symbol) {
  const Symbol *unhosted{&symbol};
  while (
      const auto *assoc{unhosted->detailsIf<semantics::HostAssocDetails>()}) {
    unhosted = &assoc->symbol();
  }
  if (const auto *binding{
          unhosted->detailsIf<semantics::ProcBindingDetails>()}) {
    if (binding->symbol().name() != symbol.name()) {
      message.Attach(binding->symbol().name(),
          "Procedure '%s' of type '%s' is bound to '%s'"_en_US, symbol.name(),
          symbol.owner().GetName().value(), binding->symbol().name());
      return &message;
    }
    unhosted = &binding->symbol();
  }
  if (const auto *use{symbol.detailsIf<semantics::UseDetails>()}) {
    message.Attach(use->location(),
        "'%s' is USE-associated with '%s' in module '%s'"_en_US, symbol.name(),
        unhosted->name(), GetUsedModule(*use).name());
  } else {
    message.Attach(
        unhosted->name(), "Declaration of '%s'"_en_US, unhosted->name());
  }
  return &message;
}

parser::Message *AttachDeclaration(
    parser::Message *message, const Symbol &symbol) {
  return message ? AttachDeclaration(*message, symbol) : nullptr;
}

class FindImpureCallHelper
    : public AnyTraverse<FindImpureCallHelper, std::optional<std::string>,
          /*TraverseAssocEntityDetails=*/false> {
  using Result = std::optional<std::string>;
  using Base = AnyTraverse<FindImpureCallHelper, Result, false>;

public:
  explicit FindImpureCallHelper(FoldingContext &c) : Base{*this}, context_{c} {}
  using Base::operator();
  Result operator()(const ProcedureRef &call) const {
    if (auto chars{characteristics::Procedure::Characterize(
            call.proc(), context_, /*emitError=*/false)}) {
      if (chars->attrs.test(characteristics::Procedure::Attr::Pure)) {
        return (*this)(call.arguments());
      }
    }
    return call.proc().GetName();
  }

private:
  FoldingContext &context_;
};

std::optional<std::string> FindImpureCall(
    FoldingContext &context, const Expr<SomeType> &expr) {
  return FindImpureCallHelper{context}(expr);
}
std::optional<std::string> FindImpureCall(
    FoldingContext &context, const ProcedureRef &proc) {
  return FindImpureCallHelper{context}(proc);
}

// Common handling for procedure pointer compatibility of left- and right-hand
// sides.  Returns nullopt if they're compatible.  Otherwise, it returns a
// message that needs to be augmented by the names of the left and right sides
// and the content of the "whyNotCompatible" string.
std::optional<parser::MessageFixedText> CheckProcCompatibility(bool isCall,
    const std::optional<characteristics::Procedure> &lhsProcedure,
    const characteristics::Procedure *rhsProcedure,
    const SpecificIntrinsic *specificIntrinsic, std::string &whyNotCompatible,
    std::optional<std::string> &warning, bool ignoreImplicitVsExplicit) {
  std::optional<parser::MessageFixedText> msg;
  if (!lhsProcedure) {
    msg = "In assignment to object %s, the target '%s' is a procedure"
          " designator"_err_en_US;
  } else if (!rhsProcedure) {
    msg = "In assignment to procedure %s, the characteristics of the target"
          " procedure '%s' could not be determined"_err_en_US;
  } else if (!isCall && lhsProcedure->functionResult &&
      rhsProcedure->functionResult &&
      !lhsProcedure->functionResult->IsCompatibleWith(
          *rhsProcedure->functionResult, &whyNotCompatible)) {
    msg =
        "Function %s associated with incompatible function designator '%s': %s"_err_en_US;
  } else if (lhsProcedure->IsCompatibleWith(*rhsProcedure,
                 ignoreImplicitVsExplicit, &whyNotCompatible, specificIntrinsic,
                 &warning)) {
    // OK
  } else if (isCall) {
    msg = "Procedure %s associated with result of reference to function '%s'"
          " that is an incompatible procedure pointer: %s"_err_en_US;
  } else if (lhsProcedure->IsPure() && !rhsProcedure->IsPure()) {
    msg = "PURE procedure %s may not be associated with non-PURE"
          " procedure designator '%s'"_err_en_US;
  } else if (lhsProcedure->IsFunction() && rhsProcedure->IsSubroutine()) {
    msg = "Function %s may not be associated with subroutine"
          " designator '%s'"_err_en_US;
  } else if (lhsProcedure->IsSubroutine() && rhsProcedure->IsFunction()) {
    msg = "Subroutine %s may not be associated with function"
          " designator '%s'"_err_en_US;
  } else if (lhsProcedure->HasExplicitInterface() &&
      !rhsProcedure->HasExplicitInterface()) {
    // Section 10.2.2.4, paragraph 3 prohibits associating a procedure pointer
    // that has an explicit interface with a procedure whose characteristics
    // don't match.  That's the case if the target procedure has an implicit
    // interface.  But this case is allowed by several other compilers as long
    // as the explicit interface can be called via an implicit interface.
    if (!lhsProcedure->CanBeCalledViaImplicitInterface()) {
      msg = "Procedure %s with explicit interface that cannot be called via "
            "an implicit interface cannot be associated with procedure "
            "designator with an implicit interface"_err_en_US;
    }
  } else if (!lhsProcedure->HasExplicitInterface() &&
      rhsProcedure->HasExplicitInterface()) {
    // OK if the target can be called via an implicit interface
    if (!rhsProcedure->CanBeCalledViaImplicitInterface() &&
        !specificIntrinsic) {
      msg = "Procedure %s with implicit interface may not be associated "
            "with procedure designator '%s' with explicit interface that "
            "cannot be called via an implicit interface"_err_en_US;
    }
  } else {
    msg = "Procedure %s associated with incompatible procedure"
          " designator '%s': %s"_err_en_US;
  }
  return msg;
}

// GetLastPointerSymbol()
static const Symbol *GetLastPointerSymbol(const Symbol &symbol) {
  return IsPointer(GetAssociationRoot(symbol)) ? &symbol : nullptr;
}
static const Symbol *GetLastPointerSymbol(const SymbolRef &symbol) {
  return GetLastPointerSymbol(*symbol);
}
static const Symbol *GetLastPointerSymbol(const Component &x) {
  const Symbol &c{x.GetLastSymbol()};
  return IsPointer(c) ? &c : GetLastPointerSymbol(x.base());
}
static const Symbol *GetLastPointerSymbol(const NamedEntity &x) {
  const auto *c{x.UnwrapComponent()};
  return c ? GetLastPointerSymbol(*c) : GetLastPointerSymbol(x.GetLastSymbol());
}
static const Symbol *GetLastPointerSymbol(const ArrayRef &x) {
  return GetLastPointerSymbol(x.base());
}
static const Symbol *GetLastPointerSymbol(const CoarrayRef &x) {
  return nullptr;
}
const Symbol *GetLastPointerSymbol(const DataRef &x) {
  return common::visit(
      [](const auto &y) { return GetLastPointerSymbol(y); }, x.u);
}

template <TypeCategory TO, TypeCategory FROM>
static std::optional<Expr<SomeType>> DataConstantConversionHelper(
    FoldingContext &context, const DynamicType &toType,
    const Expr<SomeType> &expr) {
  DynamicType sizedType{FROM, toType.kind()};
  if (auto sized{
          Fold(context, ConvertToType(sizedType, Expr<SomeType>{expr}))}) {
    if (const auto *someExpr{UnwrapExpr<Expr<SomeKind<FROM>>>(*sized)}) {
      return common::visit(
          [](const auto &w) -> std::optional<Expr<SomeType>> {
            using FromType = ResultType<decltype(w)>;
            static constexpr int kind{FromType::kind};
            if constexpr (IsValidKindOfIntrinsicType(TO, kind)) {
              if (const auto *fromConst{UnwrapExpr<Constant<FromType>>(w)}) {
                using FromWordType = typename FromType::Scalar;
                using LogicalType = value::Logical<FromWordType::bits>;
                using ElementType =
                    std::conditional_t<TO == TypeCategory::Logical, LogicalType,
                        typename LogicalType::Word>;
                std::vector<ElementType> values;
                auto at{fromConst->lbounds()};
                auto shape{fromConst->shape()};
                for (auto n{GetSize(shape)}; n-- > 0;
                     fromConst->IncrementSubscripts(at)) {
                  auto elt{fromConst->At(at)};
                  if constexpr (TO == TypeCategory::Logical) {
                    values.emplace_back(std::move(elt));
                  } else {
                    values.emplace_back(elt.word());
                  }
                }
                return {AsGenericExpr(AsExpr(Constant<Type<TO, kind>>{
                    std::move(values), std::move(shape)}))};
              }
            }
            return std::nullopt;
          },
          someExpr->u);
    }
  }
  return std::nullopt;
}

std::optional<Expr<SomeType>> DataConstantConversionExtension(
    FoldingContext &context, const DynamicType &toType,
    const Expr<SomeType> &expr0) {
  Expr<SomeType> expr{Fold(context, Expr<SomeType>{expr0})};
  if (!IsActuallyConstant(expr)) {
    return std::nullopt;
  }
  if (auto fromType{expr.GetType()}) {
    if (toType.category() == TypeCategory::Logical &&
        fromType->category() == TypeCategory::Integer) {
      return DataConstantConversionHelper<TypeCategory::Logical,
          TypeCategory::Integer>(context, toType, expr);
    }
    if (toType.category() == TypeCategory::Integer &&
        fromType->category() == TypeCategory::Logical) {
      return DataConstantConversionHelper<TypeCategory::Integer,
          TypeCategory::Logical>(context, toType, expr);
    }
  }
  return std::nullopt;
}

bool IsAllocatableOrPointerObject(const Expr<SomeType> &expr) {
  const semantics::Symbol *sym{UnwrapWholeSymbolOrComponentDataRef(expr)};
  return (sym &&
             semantics::IsAllocatableOrObjectPointer(&sym->GetUltimate())) ||
      evaluate::IsObjectPointer(expr);
}

bool IsAllocatableDesignator(const Expr<SomeType> &expr) {
  // Allocatable sub-objects are not themselves allocatable (9.5.3.1 NOTE 2).
  if (const semantics::Symbol *
      sym{UnwrapWholeSymbolOrComponentOrCoarrayRef(expr)}) {
    return semantics::IsAllocatable(sym->GetUltimate());
  }
  return false;
}

bool MayBePassedAsAbsentOptional(const Expr<SomeType> &expr) {
  const semantics::Symbol *sym{UnwrapWholeSymbolOrComponentDataRef(expr)};
  // 15.5.2.12 1. is pretty clear that an unallocated allocatable/pointer actual
  // may be passed to a non-allocatable/non-pointer optional dummy. Note that
  // other compilers (like nag, nvfortran, ifort, gfortran and xlf) seems to
  // ignore this point in intrinsic contexts (e.g CMPLX argument).
  return (sym && semantics::IsOptional(*sym)) ||
      IsAllocatableOrPointerObject(expr);
}

std::optional<Expr<SomeType>> HollerithToBOZ(FoldingContext &context,
    const Expr<SomeType> &expr, const DynamicType &type) {
  if (std::optional<std::string> chValue{GetScalarConstantValue<Ascii>(expr)}) {
    // Pad on the right with spaces when short, truncate the right if long.
    auto bytes{static_cast<std::size_t>(
        ToInt64(type.MeasureSizeInBytes(context, false)).value())};
    BOZLiteralConstant bits{0};
    for (std::size_t j{0}; j < bytes; ++j) {
      auto idx{isHostLittleEndian ? j : bytes - j - 1};
      char ch{idx >= chValue->size() ? ' ' : chValue->at(idx)};
      BOZLiteralConstant chBOZ{static_cast<unsigned char>(ch)};
      bits = bits.IOR(chBOZ.SHIFTL(8 * j));
    }
    return ConvertToType(type, Expr<SomeType>{bits});
  } else {
    return std::nullopt;
  }
}

// Extracts a whole symbol being used as a bound of a dummy argument,
// possibly wrapped with parentheses or MAX(0, ...).
// Works with any integer expression.
template <typename T> const Symbol *GetBoundSymbol(const Expr<T> &);
template <int KIND>
const Symbol *GetBoundSymbol(
    const Expr<Type<TypeCategory::Integer, KIND>> &expr) {
  using T = Type<TypeCategory::Integer, KIND>;
  return common::visit(
      common::visitors{
          [](const Extremum<T> &max) -> const Symbol * {
            if (max.ordering == Ordering::Greater) {
              if (auto zero{ToInt64(max.left())}; zero && *zero == 0) {
                return GetBoundSymbol(max.right());
              }
            }
            return nullptr;
          },
          [](const Parentheses<T> &x) { return GetBoundSymbol(x.left()); },
          [](const Designator<T> &x) -> const Symbol * {
            if (const auto *ref{std::get_if<SymbolRef>(&x.u)}) {
              return &**ref;
            }
            return nullptr;
          },
          [](const Convert<T, TypeCategory::Integer> &x) {
            return common::visit(
                [](const auto &y) -> const Symbol * {
                  using yType = std::decay_t<decltype(y)>;
                  using yResult = typename yType::Result;
                  if constexpr (yResult::kind <= KIND) {
                    return GetBoundSymbol(y);
                  } else {
                    return nullptr;
                  }
                },
                x.left().u);
          },
          [](const auto &) -> const Symbol * { return nullptr; },
      },
      expr.u);
}
template <>
const Symbol *GetBoundSymbol<SomeInteger>(const Expr<SomeInteger> &expr) {
  return common::visit(
      [](const auto &kindExpr) { return GetBoundSymbol(kindExpr); }, expr.u);
}

template <typename T>
std::optional<bool> AreEquivalentInInterface(
    const Expr<T> &x, const Expr<T> &y) {
  auto xVal{ToInt64(x)};
  auto yVal{ToInt64(y)};
  if (xVal && yVal) {
    return *xVal == *yVal;
  } else if (xVal || yVal) {
    return false;
  }
  const Symbol *xSym{GetBoundSymbol(x)};
  const Symbol *ySym{GetBoundSymbol(y)};
  if (xSym && ySym) {
    if (&xSym->GetUltimate() == &ySym->GetUltimate()) {
      return true; // USE/host associated same symbol
    }
    auto xNum{semantics::GetDummyArgumentNumber(xSym)};
    auto yNum{semantics::GetDummyArgumentNumber(ySym)};
    if (xNum && yNum) {
      if (*xNum == *yNum) {
        auto xType{DynamicType::From(*xSym)};
        auto yType{DynamicType::From(*ySym)};
        return xType && yType && xType->IsEquivalentTo(*yType);
      }
    }
    return false;
  } else if (xSym || ySym) {
    return false;
  }
  // Neither expression is an integer constant or a whole symbol.
  if (x == y) {
    return true;
  } else {
    return std::nullopt; // not sure
  }
}
template std::optional<bool> AreEquivalentInInterface<SubscriptInteger>(
    const Expr<SubscriptInteger> &, const Expr<SubscriptInteger> &);
template std::optional<bool> AreEquivalentInInterface<SomeInteger>(
    const Expr<SomeInteger> &, const Expr<SomeInteger> &);

bool CheckForCoindexedObject(parser::ContextualMessages &messages,
    const std::optional<ActualArgument> &arg, const std::string &procName,
    const std::string &argName) {
  if (arg && ExtractCoarrayRef(arg->UnwrapExpr())) {
    messages.Say(arg->sourceLocation(),
        "'%s' argument to '%s' may not be a coindexed object"_err_en_US,
        argName, procName);
    return false;
  } else {
    return true;
  }
}

} // namespace Fortran::evaluate

namespace Fortran::semantics {

const Symbol &ResolveAssociations(const Symbol &original) {
  const Symbol &symbol{original.GetUltimate()};
  if (const auto *details{symbol.detailsIf<AssocEntityDetails>()}) {
    if (!details->rank()) { // Not RANK(n) or RANK(*)
      if (const Symbol * nested{UnwrapWholeSymbolDataRef(details->expr())}) {
        return ResolveAssociations(*nested);
      }
    }
  }
  return symbol;
}

// When a construct association maps to a variable, and that variable
// is not an array with a vector-valued subscript, return the base
// Symbol of that variable, else nullptr.  Descends into other construct
// associations when one associations maps to another.
static const Symbol *GetAssociatedVariable(const AssocEntityDetails &details) {
  if (const auto &expr{details.expr()}) {
    if (IsVariable(*expr) && !HasVectorSubscript(*expr)) {
      if (const Symbol * varSymbol{GetFirstSymbol(*expr)}) {
        return &GetAssociationRoot(*varSymbol);
      }
    }
  }
  return nullptr;
}

const Symbol &GetAssociationRoot(const Symbol &original) {
  const Symbol &symbol{ResolveAssociations(original)};
  if (const auto *details{symbol.detailsIf<AssocEntityDetails>()}) {
    if (const Symbol * root{GetAssociatedVariable(*details)}) {
      return *root;
    }
  }
  return symbol;
}

const Symbol *GetMainEntry(const Symbol *symbol) {
  if (symbol) {
    if (const auto *subpDetails{symbol->detailsIf<SubprogramDetails>()}) {
      if (const Scope * scope{subpDetails->entryScope()}) {
        if (const Symbol * main{scope->symbol()}) {
          return main;
        }
      }
    }
  }
  return symbol;
}

bool IsVariableName(const Symbol &original) {
  const Symbol &ultimate{original.GetUltimate()};
  return !IsNamedConstant(ultimate) &&
      (ultimate.has<ObjectEntityDetails>() ||
          ultimate.has<AssocEntityDetails>());
}

static bool IsPureProcedureImpl(
    const Symbol &original, semantics::UnorderedSymbolSet &set) {
  // An ENTRY is pure if its containing subprogram is
  const Symbol &symbol{DEREF(GetMainEntry(&original.GetUltimate()))};
  if (set.find(symbol) != set.end()) {
    return true;
  }
  set.emplace(symbol);
  if (const auto *procDetails{symbol.detailsIf<ProcEntityDetails>()}) {
    if (procDetails->procInterface()) {
      // procedure with a pure interface
      return IsPureProcedureImpl(*procDetails->procInterface(), set);
    }
  } else if (const auto *details{symbol.detailsIf<ProcBindingDetails>()}) {
    return IsPureProcedureImpl(details->symbol(), set);
  } else if (!IsProcedure(symbol)) {
    return false;
  }
  if (IsStmtFunction(symbol)) {
    // Section 15.7(1) states that a statement function is PURE if it does not
    // reference an IMPURE procedure or a VOLATILE variable
    if (const auto &expr{symbol.get<SubprogramDetails>().stmtFunction()}) {
      for (const SymbolRef &ref : evaluate::CollectSymbols(*expr)) {
        if (&*ref == &symbol) {
          return false; // error recovery, recursion is caught elsewhere
        }
        if (IsFunction(*ref) && !IsPureProcedureImpl(*ref, set)) {
          return false;
        }
        if (ref->GetUltimate().attrs().test(Attr::VOLATILE)) {
          return false;
        }
      }
    }
    return true; // statement function was not found to be impure
  }
  return symbol.attrs().test(Attr::PURE) ||
      (symbol.attrs().test(Attr::ELEMENTAL) &&
          !symbol.attrs().test(Attr::IMPURE));
}

bool IsPureProcedure(const Symbol &original) {
  semantics::UnorderedSymbolSet set;
  return IsPureProcedureImpl(original, set);
}

bool IsPureProcedure(const Scope &scope) {
  const Symbol *symbol{scope.GetSymbol()};
  return symbol && IsPureProcedure(*symbol);
}

bool IsExplicitlyImpureProcedure(const Symbol &original) {
  // An ENTRY is IMPURE if its containing subprogram is so
  return DEREF(GetMainEntry(&original.GetUltimate()))
      .attrs()
      .test(Attr::IMPURE);
}

bool IsElementalProcedure(const Symbol &original) {
  // An ENTRY is elemental if its containing subprogram is
  const Symbol &symbol{DEREF(GetMainEntry(&original.GetUltimate()))};
  if (IsProcedure(symbol)) {
    auto &foldingContext{symbol.owner().context().foldingContext()};
    auto restorer{foldingContext.messages().DiscardMessages()};
    auto proc{evaluate::characteristics::Procedure::Characterize(
        symbol, foldingContext)};
    return proc &&
        proc->attrs.test(evaluate::characteristics::Procedure::Attr::Elemental);
  } else {
    return false;
  }
}

bool IsFunction(const Symbol &symbol) {
  const Symbol &ultimate{symbol.GetUltimate()};
  return ultimate.test(Symbol::Flag::Function) ||
      (!ultimate.test(Symbol::Flag::Subroutine) &&
          common::visit(
              common::visitors{
                  [](const SubprogramDetails &x) { return x.isFunction(); },
                  [](const ProcEntityDetails &x) {
                    const Symbol *ifc{x.procInterface()};
                    return x.type() || (ifc && IsFunction(*ifc));
                  },
                  [](const ProcBindingDetails &x) {
                    return IsFunction(x.symbol());
                  },
                  [](const auto &) { return false; },
              },
              ultimate.details()));
}

bool IsFunction(const Scope &scope) {
  const Symbol *symbol{scope.GetSymbol()};
  return symbol && IsFunction(*symbol);
}

bool IsProcedure(const Symbol &symbol) {
  return common::visit(common::visitors{
                           [&symbol](const SubprogramDetails &) {
                             const Scope *scope{symbol.scope()};
                             // Main programs & BLOCK DATA are not procedures.
                             return !scope ||
                                 scope->kind() == Scope::Kind::Subprogram;
                           },
                           [](const SubprogramNameDetails &) { return true; },
                           [](const ProcEntityDetails &) { return true; },
                           [](const GenericDetails &) { return true; },
                           [](const ProcBindingDetails &) { return true; },
                           [](const auto &) { return false; },
                       },
      symbol.GetUltimate().details());
}

bool IsProcedure(const Scope &scope) {
  const Symbol *symbol{scope.GetSymbol()};
  return symbol && IsProcedure(*symbol);
}

bool IsProcedurePointer(const Symbol &original) {
  const Symbol &symbol{GetAssociationRoot(original)};
  return IsPointer(symbol) && IsProcedure(symbol);
}

bool IsProcedurePointer(const Symbol *symbol) {
  return symbol && IsProcedurePointer(*symbol);
}

bool IsObjectPointer(const Symbol *original) {
  if (original) {
    const Symbol &symbol{GetAssociationRoot(*original)};
    return IsPointer(symbol) && !IsProcedure(symbol);
  } else {
    return false;
  }
}

bool IsAllocatableOrObjectPointer(const Symbol *original) {
  if (original) {
    const Symbol &ultimate{original->GetUltimate()};
    if (const auto *assoc{ultimate.detailsIf<AssocEntityDetails>()}) {
      // Only SELECT RANK construct entities can be ALLOCATABLE/POINTER.
      return (assoc->rank() || assoc->IsAssumedSize() ||
                 assoc->IsAssumedRank()) &&
          IsAllocatableOrObjectPointer(UnwrapWholeSymbolDataRef(assoc->expr()));
    } else {
      return IsAllocatable(ultimate) ||
          (IsPointer(ultimate) && !IsProcedure(ultimate));
    }
  } else {
    return false;
  }
}

const Symbol *FindCommonBlockContaining(const Symbol &original) {
  const Symbol &root{GetAssociationRoot(original)};
  const auto *details{root.detailsIf<ObjectEntityDetails>()};
  return details ? details->commonBlock() : nullptr;
}

// 3.11 automatic data object
bool IsAutomatic(const Symbol &original) {
  const Symbol &symbol{original.GetUltimate()};
  if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
    if (!object->isDummy() && !IsAllocatable(symbol) && !IsPointer(symbol)) {
      if (const DeclTypeSpec * type{symbol.GetType()}) {
        // If a type parameter value is not a constant expression, the
        // object is automatic.
        if (type->category() == DeclTypeSpec::Character) {
          if (const auto &length{
                  type->characterTypeSpec().length().GetExplicit()}) {
            if (!evaluate::IsConstantExpr(*length)) {
              return true;
            }
          }
        } else if (const DerivedTypeSpec * derived{type->AsDerived()}) {
          for (const auto &pair : derived->parameters()) {
            if (const auto &value{pair.second.GetExplicit()}) {
              if (!evaluate::IsConstantExpr(*value)) {
                return true;
              }
            }
          }
        }
      }
      // If an array bound is not a constant expression, the object is
      // automatic.
      for (const ShapeSpec &dim : object->shape()) {
        if (const auto &lb{dim.lbound().GetExplicit()}) {
          if (!evaluate::IsConstantExpr(*lb)) {
            return true;
          }
        }
        if (const auto &ub{dim.ubound().GetExplicit()}) {
          if (!evaluate::IsConstantExpr(*ub)) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

bool IsSaved(const Symbol &original) {
  const Symbol &symbol{GetAssociationRoot(original)};
  const Scope &scope{symbol.owner()};
  const common::LanguageFeatureControl &features{
      scope.context().languageFeatures()};
  auto scopeKind{scope.kind()};
  if (symbol.has<AssocEntityDetails>()) {
    return false; // ASSOCIATE(non-variable)
  } else if (scopeKind == Scope::Kind::DerivedType) {
    return false; // this is a component
  } else if (symbol.attrs().test(Attr::SAVE)) {
    return true; // explicit SAVE attribute
  } else if (IsDummy(symbol) || IsFunctionResult(symbol) ||
      IsAutomatic(symbol) || IsNamedConstant(symbol)) {
    return false;
  } else if (scopeKind == Scope::Kind::Module ||
      (scopeKind == Scope::Kind::MainProgram &&
          (symbol.attrs().test(Attr::TARGET) || evaluate::IsCoarray(symbol)) &&
          Fortran::evaluate::CanCUDASymbolHaveSaveAttr(symbol))) {
    // 8.5.16p4
    // In main programs, implied SAVE matters only for pointer
    // initialization targets and coarrays.
    return true;
  } else if (scopeKind == Scope::Kind::MainProgram &&
      (features.IsEnabled(common::LanguageFeature::SaveMainProgram) ||
          (features.IsEnabled(
               common::LanguageFeature::SaveBigMainProgramVariables) &&
              symbol.size() > 32)) &&
      Fortran::evaluate::CanCUDASymbolHaveSaveAttr(symbol)) {
    // With SaveBigMainProgramVariables, keeping all unsaved main program
    // variables of 32 bytes or less on the stack allows keeping numerical and
    // logical scalars, small scalar characters or derived, small arrays, and
    // scalar descriptors on the stack. This leaves more room for lower level
    // optimizers to do register promotion or get easy aliasing information.
    return true;
  } else if (features.IsEnabled(common::LanguageFeature::DefaultSave) &&
      (scopeKind == Scope::Kind::MainProgram ||
          (scope.kind() == Scope::Kind::Subprogram &&
              !(scope.symbol() &&
                  scope.symbol()->attrs().test(Attr::RECURSIVE))))) {
    // -fno-automatic/-save/-Msave option applies to all objects in executable
    // main programs and subprograms unless they are explicitly RECURSIVE.
    return true;
  } else if (symbol.test(Symbol::Flag::InDataStmt)) {
    return true;
  } else if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()};
             object && object->init()) {
    return true;
  } else if (IsProcedurePointer(symbol) && symbol.has<ProcEntityDetails>() &&
      symbol.get<ProcEntityDetails>().init()) {
    return true;
  } else if (scope.hasSAVE()) {
    return true; // bare SAVE statement
  } else if (const Symbol * block{FindCommonBlockContaining(symbol)};
             block && block->attrs().test(Attr::SAVE)) {
    return true; // in COMMON with SAVE
  } else {
    return false;
  }
}

bool IsDummy(const Symbol &symbol) {
  return common::visit(
      common::visitors{[](const EntityDetails &x) { return x.isDummy(); },
          [](const ObjectEntityDetails &x) { return x.isDummy(); },
          [](const ProcEntityDetails &x) { return x.isDummy(); },
          [](const SubprogramDetails &x) { return x.isDummy(); },
          [](const auto &) { return false; }},
      ResolveAssociations(symbol).details());
}

bool IsAssumedShape(const Symbol &symbol) {
  const Symbol &ultimate{ResolveAssociations(symbol)};
  const auto *object{ultimate.detailsIf<ObjectEntityDetails>()};
  return object && object->IsAssumedShape() &&
      !semantics::IsAllocatableOrObjectPointer(&ultimate);
}

bool IsDeferredShape(const Symbol &symbol) {
  const Symbol &ultimate{ResolveAssociations(symbol)};
  const auto *object{ultimate.detailsIf<ObjectEntityDetails>()};
  return object && object->CanBeDeferredShape() &&
      semantics::IsAllocatableOrObjectPointer(&ultimate);
}

bool IsFunctionResult(const Symbol &original) {
  const Symbol &symbol{GetAssociationRoot(original)};
  return common::visit(
      common::visitors{
          [](const EntityDetails &x) { return x.isFuncResult(); },
          [](const ObjectEntityDetails &x) { return x.isFuncResult(); },
          [](const ProcEntityDetails &x) { return x.isFuncResult(); },
          [](const auto &) { return false; },
      },
      symbol.details());
}

bool IsKindTypeParameter(const Symbol &symbol) {
  const auto *param{symbol.GetUltimate().detailsIf<TypeParamDetails>()};
  return param && param->attr() == common::TypeParamAttr::Kind;
}

bool IsLenTypeParameter(const Symbol &symbol) {
  const auto *param{symbol.GetUltimate().detailsIf<TypeParamDetails>()};
  return param && param->attr() == common::TypeParamAttr::Len;
}

bool IsExtensibleType(const DerivedTypeSpec *derived) {
  return !IsSequenceOrBindCType(derived) && !IsIsoCType(derived);
}

bool IsSequenceOrBindCType(const DerivedTypeSpec *derived) {
  return derived &&
      (derived->typeSymbol().attrs().test(Attr::BIND_C) ||
          derived->typeSymbol().get<DerivedTypeDetails>().sequence());
}

static bool IsSameModule(const Scope *x, const Scope *y) {
  if (x == y) {
    return true;
  } else if (x && y) {
    // Allow for a builtin module to be read from distinct paths
    const Symbol *xSym{x->symbol()};
    const Symbol *ySym{y->symbol()};
    if (xSym && ySym && xSym->name() == ySym->name()) {
      const auto *xMod{xSym->detailsIf<ModuleDetails>()};
      const auto *yMod{ySym->detailsIf<ModuleDetails>()};
      if (xMod && yMod) {
        auto xHash{xMod->moduleFileHash()};
        auto yHash{yMod->moduleFileHash()};
        return xHash && yHash && *xHash == *yHash;
      }
    }
  }
  return false;
}

bool IsBuiltinDerivedType(const DerivedTypeSpec *derived, const char *name) {
  if (derived) {
    const auto &symbol{derived->typeSymbol()};
    const Scope &scope{symbol.owner()};
    return symbol.name() == "__builtin_"s + name &&
        IsSameModule(&scope, scope.context().GetBuiltinsScope());
  } else {
    return false;
  }
}

bool IsBuiltinCPtr(const Symbol &symbol) {
  if (const DeclTypeSpec *declType = symbol.GetType()) {
    if (const DerivedTypeSpec *derived = declType->AsDerived()) {
      return IsIsoCType(derived);
    }
  }
  return false;
}

bool IsIsoCType(const DerivedTypeSpec *derived) {
  return IsBuiltinDerivedType(derived, "c_ptr") ||
      IsBuiltinDerivedType(derived, "c_funptr");
}

bool IsEventType(const DerivedTypeSpec *derived) {
  return IsBuiltinDerivedType(derived, "event_type");
}

bool IsLockType(const DerivedTypeSpec *derived) {
  return IsBuiltinDerivedType(derived, "lock_type");
}

bool IsNotifyType(const DerivedTypeSpec *derived) {
  return IsBuiltinDerivedType(derived, "notify_type");
}

bool IsIeeeFlagType(const DerivedTypeSpec *derived) {
  return IsBuiltinDerivedType(derived, "ieee_flag_type");
}

bool IsIeeeRoundType(const DerivedTypeSpec *derived) {
  return IsBuiltinDerivedType(derived, "ieee_round_type");
}

bool IsTeamType(const DerivedTypeSpec *derived) {
  return IsBuiltinDerivedType(derived, "team_type");
}

bool IsBadCoarrayType(const DerivedTypeSpec *derived) {
  return IsTeamType(derived) || IsIsoCType(derived);
}

bool IsEventTypeOrLockType(const DerivedTypeSpec *derivedTypeSpec) {
  return IsEventType(derivedTypeSpec) || IsLockType(derivedTypeSpec);
}

int CountLenParameters(const DerivedTypeSpec &type) {
  return llvm::count_if(
      type.parameters(), [](const auto &pair) { return pair.second.isLen(); });
}

int CountNonConstantLenParameters(const DerivedTypeSpec &type) {
  return llvm::count_if(type.parameters(), [](const auto &pair) {
    if (!pair.second.isLen()) {
      return false;
    } else if (const auto &expr{pair.second.GetExplicit()}) {
      return !IsConstantExpr(*expr);
    } else {
      return true;
    }
  });
}

const Symbol &GetUsedModule(const UseDetails &details) {
  return DEREF(details.symbol().owner().symbol());
}

static const Symbol *FindFunctionResult(
    const Symbol &original, UnorderedSymbolSet &seen) {
  const Symbol &root{GetAssociationRoot(original)};
  ;
  if (!seen.insert(root).second) {
    return nullptr; // don't loop
  }
  return common::visit(
      common::visitors{[](const SubprogramDetails &subp) {
                         return subp.isFunction() ? &subp.result() : nullptr;
                       },
          [&](const ProcEntityDetails &proc) {
            const Symbol *iface{proc.procInterface()};
            return iface ? FindFunctionResult(*iface, seen) : nullptr;
          },
          [&](const ProcBindingDetails &binding) {
            return FindFunctionResult(binding.symbol(), seen);
          },
          [](const auto &) -> const Symbol * { return nullptr; }},
      root.details());
}

const Symbol *FindFunctionResult(const Symbol &symbol) {
  UnorderedSymbolSet seen;
  return FindFunctionResult(symbol, seen);
}

// These are here in Evaluate/tools.cpp so that Evaluate can use
// them; they cannot be defined in symbol.h due to the dependence
// on Scope.

bool SymbolSourcePositionCompare::operator()(
    const SymbolRef &x, const SymbolRef &y) const {
  return x->GetSemanticsContext().allCookedSources().Precedes(
      x->name(), y->name());
}
bool SymbolSourcePositionCompare::operator()(
    const MutableSymbolRef &x, const MutableSymbolRef &y) const {
  return x->GetSemanticsContext().allCookedSources().Precedes(
      x->name(), y->name());
}

SemanticsContext &Symbol::GetSemanticsContext() const {
  return DEREF(owner_).context();
}

bool AreTkCompatibleTypes(const DeclTypeSpec *x, const DeclTypeSpec *y) {
  if (x && y) {
    if (auto xDt{evaluate::DynamicType::From(*x)}) {
      if (auto yDt{evaluate::DynamicType::From(*y)}) {
        return xDt->IsTkCompatibleWith(*yDt);
      }
    }
  }
  return false;
}

common::IgnoreTKRSet GetIgnoreTKR(const Symbol &symbol) {
  common::IgnoreTKRSet result;
  if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
    result = object->ignoreTKR();
    if (const Symbol * ownerSymbol{symbol.owner().symbol()}) {
      if (const auto *ownerSubp{ownerSymbol->detailsIf<SubprogramDetails>()}) {
        if (ownerSubp->defaultIgnoreTKR()) {
          result |= common::ignoreTKRAll;
        }
      }
    }
  }
  return result;
}

std::optional<int> GetDummyArgumentNumber(const Symbol *symbol) {
  if (symbol) {
    if (IsDummy(*symbol)) {
      if (const Symbol * subpSym{symbol->owner().symbol()}) {
        if (const auto *subp{subpSym->detailsIf<SubprogramDetails>()}) {
          int j{0};
          for (const Symbol *dummy : subp->dummyArgs()) {
            if (dummy == symbol) {
              return j;
            }
            ++j;
          }
        }
      }
    }
  }
  return std::nullopt;
}

// Given a symbol that is a SubprogramNameDetails in a submodule, try to
// find its interface definition in its module or ancestor submodule.
const Symbol *FindAncestorModuleProcedure(const Symbol *symInSubmodule) {
  if (symInSubmodule && symInSubmodule->owner().IsSubmodule()) {
    if (const auto *nameDetails{
            symInSubmodule->detailsIf<semantics::SubprogramNameDetails>()};
        nameDetails &&
        nameDetails->kind() == semantics::SubprogramKind::Module) {
      const Symbol *next{symInSubmodule->owner().symbol()};
      while (const Symbol * submodSym{next}) {
        next = nullptr;
        if (const auto *modDetails{
                submodSym->detailsIf<semantics::ModuleDetails>()};
            modDetails && modDetails->isSubmodule() && modDetails->scope()) {
          if (const semantics::Scope & parent{modDetails->scope()->parent()};
              parent.IsSubmodule() || parent.IsModule()) {
            if (auto iter{parent.find(symInSubmodule->name())};
                iter != parent.end()) {
              const Symbol &proc{iter->second->GetUltimate()};
              if (IsProcedure(proc)) {
                return &proc;
              }
            } else if (parent.IsSubmodule()) {
              next = parent.symbol();
            }
          }
        }
      }
    }
  }
  return nullptr;
}

} // namespace Fortran::semantics
