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
  const Symbol &symbol{ref.GetLastSymbol()};
  if (auto dyType{DynamicType::From(symbol)}) {
    return TypedWrapper<Designator, DataRef>(*dyType, std::move(ref));
  }
  return std::nullopt;
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
    const ActualArgument &arg, bool intoSubstring) {
  if (const Expr<SomeType> *expr{arg.UnwrapExpr()}) {
    return ExtractDataRef(*expr, intoSubstring);
  } else {
    return std::nullopt;
  }
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
  const Symbol &root{GetAssociationRoot(symbol)};
  return !IsNamedConstant(root) && root.has<semantics::ObjectEntityDetails>();
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
          [&](BOZLiteralConstant &&bx,
              Expr<SomeInteger> &&iy) -> ConvertRealOperandsResult {
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
            messages.Say("operands must be INTEGER or REAL"_err_en_US);
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
  }
  return NoExpr();
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

Expr<SomeReal> GetComplexPart(const Expr<SomeComplex> &z, bool isImaginary) {
  return common::visit(
      [&](const auto &zk) {
        static constexpr int kind{ResultType<decltype(zk)>::kind};
        return AsCategoryExpr(ComplexComponent<kind>{isImaginary, zk});
      },
      z.u);
}

Expr<SomeReal> GetComplexPart(Expr<SomeComplex> &&z, bool isImaginary) {
  return common::visit(
      [&](auto &&zk) {
        static constexpr int kind{ResultType<decltype(zk)>::kind};
        return AsCategoryExpr(
            ComplexComponent<kind>{isImaginary, std::move(zk)});
      },
      z.u);
}

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
    parser::ContextualMessages &messages, Expr<SomeComplex> &&zx,
    Expr<SomeKind<RCAT>> &&iry, [[maybe_unused]] int defaultRealKind) {
  Expr<SomeReal> zr{GetComplexPart(zx, false)};
  Expr<SomeReal> zi{GetComplexPart(zx, true)};
  if constexpr (std::is_same_v<OPR<LargestReal>, Add<LargestReal>> ||
      std::is_same_v<OPR<LargestReal>, Subtract<LargestReal>>) {
    // (a,b) + x -> (a+x, b)
    // (a,b) - x -> (a-x, b)
    if (std::optional<Expr<SomeType>> rr{
            NumericOperation<OPR>(messages, AsGenericExpr(std::move(zr)),
                AsGenericExpr(std::move(iry)), defaultRealKind)}) {
      return Package(ConstructComplex(messages, std::move(*rr),
          AsGenericExpr(std::move(zi)), defaultRealKind));
    }
  } else if constexpr (allowOperandDuplication &&
      (std::is_same_v<OPR<LargestReal>, Multiply<LargestReal>> ||
          std::is_same_v<OPR<LargestReal>, Divide<LargestReal>>)) {
    // (a,b) * x -> (a*x, b*x)
    // (a,b) / x -> (a/x, b/x)
    auto copy{iry};
    auto rr{NumericOperation<OPR>(messages, AsGenericExpr(std::move(zr)),
        AsGenericExpr(std::move(iry)), defaultRealKind)};
    auto ri{NumericOperation<OPR>(messages, AsGenericExpr(std::move(zi)),
        AsGenericExpr(std::move(copy)), defaultRealKind)};
    if (auto parts{common::AllPresent(std::move(rr), std::move(ri))}) {
      return Package(ConstructComplex(messages, std::get<0>(std::move(*parts)),
          std::get<1>(std::move(*parts)), defaultRealKind));
    }
  } else if constexpr (RCAT == TypeCategory::Integer &&
      std::is_same_v<OPR<LargestReal>, Power<LargestReal>>) {
    // COMPLEX**INTEGER is a special case that doesn't convert the exponent.
    static_assert(RCAT == TypeCategory::Integer);
    return Package(common::visit(
        [&](auto &&zxk) {
          using Ty = ResultType<decltype(zxk)>;
          return AsCategoryExpr(
              AsExpr(RealToIntPower<Ty>{std::move(zxk), std::move(iry)}));
        },
        std::move(zx.u)));
  } else {
    // (a,b) ** x -> (a,b) ** (x,0)
    if constexpr (RCAT == TypeCategory::Integer) {
      Expr<SomeComplex> zy{ConvertTo(zx, std::move(iry))};
      return Package(PromoteAndCombine<OPR>(std::move(zx), std::move(zy)));
    } else {
      Expr<SomeComplex> zy{PromoteRealToComplex(std::move(iry))};
      return Package(PromoteAndCombine<OPR>(std::move(zx), std::move(zy)));
    }
  }
  return NoExpr();
}

// Mixed COMPLEX operations with the COMPLEX operand on the right.
//  x + (a,b) -> (x+a, b)
//  x - (a,b) -> (x-a, -b)
//  x * (a,b) -> (x*a, x*b)
//  x / (a,b) -> (x,0) / (a,b)   (and **)
template <template <typename> class OPR, TypeCategory LCAT>
std::optional<Expr<SomeType>> MixedComplexRight(
    parser::ContextualMessages &messages, Expr<SomeKind<LCAT>> &&irx,
    Expr<SomeComplex> &&zy, [[maybe_unused]] int defaultRealKind) {
  if constexpr (std::is_same_v<OPR<LargestReal>, Add<LargestReal>>) {
    // x + (a,b) -> (a,b) + x -> (a+x, b)
    return MixedComplexLeft<OPR, LCAT>(
        messages, std::move(zy), std::move(irx), defaultRealKind);
  } else if constexpr (allowOperandDuplication &&
      std::is_same_v<OPR<LargestReal>, Multiply<LargestReal>>) {
    // x * (a,b) -> (a,b) * x -> (a*x, b*x)
    return MixedComplexLeft<OPR, LCAT>(
        messages, std::move(zy), std::move(irx), defaultRealKind);
  } else if constexpr (std::is_same_v<OPR<LargestReal>,
                           Subtract<LargestReal>>) {
    // x - (a,b) -> (x-a, -b)
    Expr<SomeReal> zr{GetComplexPart(zy, false)};
    Expr<SomeReal> zi{GetComplexPart(zy, true)};
    if (std::optional<Expr<SomeType>> rr{
            NumericOperation<Subtract>(messages, AsGenericExpr(std::move(irx)),
                AsGenericExpr(std::move(zr)), defaultRealKind)}) {
      return Package(ConstructComplex(messages, std::move(*rr),
          AsGenericExpr(-std::move(zi)), defaultRealKind));
    }
  } else {
    // x / (a,b) -> (x,0) / (a,b)
    if constexpr (LCAT == TypeCategory::Integer) {
      Expr<SomeComplex> zx{ConvertTo(zy, std::move(irx))};
      return Package(PromoteAndCombine<OPR>(std::move(zx), std::move(zy)));
    } else {
      Expr<SomeComplex> zx{PromoteRealToComplex(std::move(irx))};
      return Package(PromoteAndCombine<OPR>(std::move(zx), std::move(zy)));
    }
  }
  return NoExpr();
}

// N.B. When a "typeless" BOZ literal constant appears as one (not both!) of
// the operands to a dyadic operation where one is permitted, it assumes the
// type and kind of the other operand.
template <template <typename> class OPR>
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
            return MixedComplexLeft<OPR>(
                messages, std::move(zx), std::move(iy), defaultRealKind);
          },
          [&](Expr<SomeComplex> &&zx, Expr<SomeReal> &&ry) {
            return MixedComplexLeft<OPR>(
                messages, std::move(zx), std::move(ry), defaultRealKind);
          },
          [&](Expr<SomeInteger> &&ix, Expr<SomeComplex> &&zy) {
            return MixedComplexRight<OPR>(
                messages, std::move(ix), std::move(zy), defaultRealKind);
          },
          [&](Expr<SomeReal> &&rx, Expr<SomeComplex> &&zy) {
            return MixedComplexRight<OPR>(
                messages, std::move(rx), std::move(zy), defaultRealKind);
          },
          // Operations with one typeless operand
          [&](BOZLiteralConstant &&bx, Expr<SomeInteger> &&iy) {
            return NumericOperation<OPR>(messages,
                AsGenericExpr(ConvertTo(iy, std::move(bx))), std::move(y),
                defaultRealKind);
          },
          [&](BOZLiteralConstant &&bx, Expr<SomeReal> &&ry) {
            return NumericOperation<OPR>(messages,
                AsGenericExpr(ConvertTo(ry, std::move(bx))), std::move(y),
                defaultRealKind);
          },
          [&](Expr<SomeInteger> &&ix, BOZLiteralConstant &&by) {
            return NumericOperation<OPR>(messages, std::move(x),
                AsGenericExpr(ConvertTo(ix, std::move(by))), defaultRealKind);
          },
          [&](Expr<SomeReal> &&rx, BOZLiteralConstant &&by) {
            return NumericOperation<OPR>(messages, std::move(x),
                AsGenericExpr(ConvertTo(rx, std::move(by))), defaultRealKind);
          },
          // Default case
          [&](auto &&, auto &&) {
            // TODO: defined operator
            messages.Say("non-numeric operands to numeric operation"_err_en_US);
            return NoExpr();
          },
      },
      std::move(x.u), std::move(y.u));
}

template std::optional<Expr<SomeType>> NumericOperation<Power>(
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
            // TODO: defined operator
            messages.Say("CHARACTER cannot be negated"_err_en_US);
            return NoExpr();
          },
          [&](Expr<SomeLogical> &&) {
            // TODO: defined operator
            messages.Say("LOGICAL cannot be negated"_err_en_US);
            return NoExpr();
          },
          [&](Expr<SomeDerived> &&) {
            // TODO: defined operator
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
              using Ty = std::decay_t<decltype(x)>;
              using CharacterType = typename Ty::Result;
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
      return false; // in SELECT RANK case
    }
  }
  const Symbol &symbol{semantics::ResolveAssociations(original)};
  if (const auto *details{symbol.detailsIf<semantics::ObjectEntityDetails>()}) {
    return details->IsAssumedRank();
  } else {
    return false;
  }
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

bool IsCoarray(const ActualArgument &arg) {
  const auto *expr{arg.UnwrapExpr()};
  return expr && IsCoarray(*expr);
}

bool IsCoarray(const Symbol &symbol) {
  return GetAssociationRoot(symbol).Corank() > 0;
}

bool IsProcedure(const Expr<SomeType> &expr) {
  return std::holds_alternative<ProcedureDesignator>(expr.u);
}
bool IsFunction(const Expr<SomeType> &expr) {
  const auto *designator{std::get_if<ProcedureDesignator>(&expr.u)};
  return designator && designator->GetType().has_value();
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

template <typename A> inline const ProcedureRef *UnwrapProcedureRef(const A &) {
  return nullptr;
}

template <typename T>
inline const ProcedureRef *UnwrapProcedureRef(const FunctionRef<T> &func) {
  return &func;
}

template <typename T>
inline const ProcedureRef *UnwrapProcedureRef(const Expr<T> &expr) {
  return common::visit(
      [](const auto &x) { return UnwrapProcedureRef(x); }, expr.u);
}

// IsObjectPointer()
bool IsObjectPointer(const Expr<SomeType> &expr, FoldingContext &context) {
  if (IsNullPointer(expr)) {
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

bool IsBareNullPointer(const Expr<SomeType> *expr) {
  return expr && std::holds_alternative<NullPointer>(expr->u);
}

// IsNullPointer()
struct IsNullPointerHelper {
  template <typename A> bool operator()(const A &) const { return false; }
  template <typename T> bool operator()(const FunctionRef<T> &call) const {
    const auto *intrinsic{call.proc().GetSpecificIntrinsic()};
    return intrinsic &&
        intrinsic->characteristics.value().attrs.test(
            characteristics::Procedure::Attr::NullPointer);
  }
  bool operator()(const NullPointer &) const { return true; }
  template <typename T> bool operator()(const Parentheses<T> &x) const {
    return (*this)(x.left());
  }
  template <typename T> bool operator()(const Expr<T> &x) const {
    return common::visit(*this, x.u);
  }
};

bool IsNullPointer(const Expr<SomeType> &expr) {
  return IsNullPointerHelper{}(expr);
}

// GetSymbolVector()
auto GetSymbolVectorHelper::operator()(const Symbol &x) const -> Result {
  if (const auto *details{x.detailsIf<semantics::AssocEntityDetails>()}) {
    return (*this)(details->expr());
  } else {
    return {x.GetUltimate()};
  }
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

// HasVectorSubscript()
struct HasVectorSubscriptHelper : public AnyTraverse<HasVectorSubscriptHelper> {
  using Base = AnyTraverse<HasVectorSubscriptHelper>;
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
    : public AnyTraverse<FindImpureCallHelper, std::optional<std::string>> {
  using Result = std::optional<std::string>;
  using Base = AnyTraverse<FindImpureCallHelper, Result>;

public:
  explicit FindImpureCallHelper(FoldingContext &c) : Base{*this}, context_{c} {}
  using Base::operator();
  Result operator()(const ProcedureRef &call) const {
    if (auto chars{
            characteristics::Procedure::Characterize(call.proc(), context_)}) {
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

// Compare procedure characteristics for equality except that rhs may be
// Pure or Elemental when lhs is not.
static bool CharacteristicsMatch(const characteristics::Procedure &lhs,
    const characteristics::Procedure &rhs) {
  using Attr = characteristics::Procedure::Attr;
  auto lhsAttrs{lhs.attrs};
  lhsAttrs.set(
      Attr::Pure, lhs.attrs.test(Attr::Pure) || rhs.attrs.test(Attr::Pure));
  lhsAttrs.set(Attr::Elemental,
      lhs.attrs.test(Attr::Elemental) || rhs.attrs.test(Attr::Elemental));
  return lhsAttrs == rhs.attrs && lhs.functionResult == rhs.functionResult &&
      lhs.dummyArguments == rhs.dummyArguments;
}

// Common handling for procedure pointer compatibility of left- and right-hand
// sides.  Returns nullopt if they're compatible.  Otherwise, it returns a
// message that needs to be augmented by the names of the left and right sides
std::optional<parser::MessageFixedText> CheckProcCompatibility(bool isCall,
    const std::optional<characteristics::Procedure> &lhsProcedure,
    const characteristics::Procedure *rhsProcedure) {
  std::optional<parser::MessageFixedText> msg;
  if (!lhsProcedure) {
    msg = "In assignment to object %s, the target '%s' is a procedure"
          " designator"_err_en_US;
  } else if (!rhsProcedure) {
    msg = "In assignment to procedure %s, the characteristics of the target"
          " procedure '%s' could not be determined"_err_en_US;
  } else if (CharacteristicsMatch(*lhsProcedure, *rhsProcedure)) {
    // OK
  } else if (isCall) {
    msg = "Procedure %s associated with result of reference to function '%s'"
          " that is an incompatible procedure pointer"_err_en_US;
  } else if (lhsProcedure->IsPure() && !rhsProcedure->IsPure()) {
    msg = "PURE procedure %s may not be associated with non-PURE"
          " procedure designator '%s'"_err_en_US;
  } else if (lhsProcedure->IsFunction() && !rhsProcedure->IsFunction()) {
    msg = "Function %s may not be associated with subroutine"
          " designator '%s'"_err_en_US;
  } else if (!lhsProcedure->IsFunction() && rhsProcedure->IsFunction()) {
    msg = "Subroutine %s may not be associated with function"
          " designator '%s'"_err_en_US;
  } else if (lhsProcedure->HasExplicitInterface() &&
      !rhsProcedure->HasExplicitInterface()) {
    // Section 10.2.2.4, paragraph 3 prohibits associating a procedure pointer
    // with an explicit interface with a procedure whose characteristics don't
    // match.  That's the case if the target procedure has an implicit
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
    if (!rhsProcedure->CanBeCalledViaImplicitInterface()) {
      msg = "Procedure %s with implicit interface may not be associated "
            "with procedure designator '%s' with explicit interface that "
            "cannot be called via an implicit interface"_err_en_US;
    }
  } else {
    msg = "Procedure %s associated with incompatible procedure"
          " designator '%s'"_err_en_US;
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
            using FromType = typename std::decay_t<decltype(w)>::Result;
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

bool IsAllocatableOrPointerObject(
    const Expr<SomeType> &expr, FoldingContext &context) {
  const semantics::Symbol *sym{UnwrapWholeSymbolOrComponentDataRef(expr)};
  return (sym && semantics::IsAllocatableOrPointer(*sym)) ||
      evaluate::IsObjectPointer(expr, context);
}

bool IsAllocatableDesignator(const Expr<SomeType> &expr) {
  // Allocatable sub-objects are not themselves allocatable (9.5.3.1 NOTE 2).
  if (const semantics::Symbol *
      sym{UnwrapWholeSymbolOrComponentOrCoarrayRef(expr)}) {
    return semantics::IsAllocatable(*sym);
  }
  return false;
}

bool MayBePassedAsAbsentOptional(
    const Expr<SomeType> &expr, FoldingContext &context) {
  const semantics::Symbol *sym{UnwrapWholeSymbolOrComponentDataRef(expr)};
  // 15.5.2.12 1. is pretty clear that an unallocated allocatable/pointer actual
  // may be passed to a non-allocatable/non-pointer optional dummy. Note that
  // other compilers (like nag, nvfortran, ifort, gfortran and xlf) seems to
  // ignore this point in intrinsic contexts (e.g CMPLX argument).
  return (sym && semantics::IsOptional(*sym)) ||
      IsAllocatableOrPointerObject(expr, context);
}

} // namespace Fortran::evaluate

namespace Fortran::semantics {

const Symbol &ResolveAssociations(const Symbol &original) {
  const Symbol &symbol{original.GetUltimate()};
  if (const auto *details{symbol.detailsIf<AssocEntityDetails>()}) {
    if (const Symbol * nested{UnwrapWholeSymbolDataRef(details->expr())}) {
      return ResolveAssociations(*nested);
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
  const Symbol &symbol{ResolveAssociations(original)};
  if (symbol.has<ObjectEntityDetails>()) {
    return !IsNamedConstant(symbol);
  } else if (const auto *assoc{symbol.detailsIf<AssocEntityDetails>()}) {
    const auto &expr{assoc->expr()};
    return expr && IsVariable(*expr) && !HasVectorSubscript(*expr);
  } else {
    return false;
  }
}

bool IsPureProcedure(const Symbol &original) {
  // An ENTRY is pure if its containing subprogram is
  const Symbol &symbol{DEREF(GetMainEntry(&original.GetUltimate()))};
  if (const auto *procDetails{symbol.detailsIf<ProcEntityDetails>()}) {
    if (const Symbol * procInterface{procDetails->interface().symbol()}) {
      // procedure component with a pure interface
      return IsPureProcedure(*procInterface);
    }
  } else if (const auto *details{symbol.detailsIf<ProcBindingDetails>()}) {
    return IsPureProcedure(details->symbol());
  } else if (!IsProcedure(symbol)) {
    return false;
  }
  if (IsStmtFunction(symbol)) {
    // Section 15.7(1) states that a statement function is PURE if it does not
    // reference an IMPURE procedure or a VOLATILE variable
    if (const auto &expr{symbol.get<SubprogramDetails>().stmtFunction()}) {
      for (const SymbolRef &ref : evaluate::CollectSymbols(*expr)) {
        if (IsFunction(*ref) && !IsPureProcedure(*ref)) {
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

bool IsPureProcedure(const Scope &scope) {
  const Symbol *symbol{scope.GetSymbol()};
  return symbol && IsPureProcedure(*symbol);
}

bool IsFunction(const Symbol &symbol) {
  const Symbol &ultimate{symbol.GetUltimate()};
  return ultimate.test(Symbol::Flag::Function) ||
      (!ultimate.test(Symbol::Flag::Subroutine) &&
          common::visit(
              common::visitors{
                  [](const SubprogramDetails &x) { return x.isFunction(); },
                  [](const ProcEntityDetails &x) {
                    const auto &ifc{x.interface()};
                    return ifc.type() ||
                        (ifc.symbol() && IsFunction(*ifc.symbol()));
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
                           [](const SubprogramDetails &) { return true; },
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

const Symbol *FindCommonBlockContaining(const Symbol &original) {
  const Symbol &root{GetAssociationRoot(original)};
  const auto *details{root.detailsIf<ObjectEntityDetails>()};
  return details ? details->commonBlock() : nullptr;
}

bool IsProcedurePointer(const Symbol &original) {
  const Symbol &symbol{GetAssociationRoot(original)};
  return IsPointer(symbol) &&
      (symbol.has<ProcEntityDetails>() || symbol.has<SubprogramDetails>());
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
          (symbol.attrs().test(Attr::TARGET) || evaluate::IsCoarray(symbol)))) {
    // 8.5.16p4
    // In main programs, implied SAVE matters only for pointer
    // initialization targets and coarrays.
    // BLOCK DATA entities must all be in COMMON,
    // which was checked above.
    return true;
  } else if (scope.context().languageFeatures().IsEnabled(
                 common::LanguageFeature::DefaultSave) &&
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
  } else if (IsProcedurePointer(symbol) &&
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
  return object && object->CanBeAssumedShape() &&
      !evaluate::IsAllocatableOrPointer(ultimate);
}

bool IsDeferredShape(const Symbol &symbol) {
  const Symbol &ultimate{ResolveAssociations(symbol)};
  const auto *object{ultimate.detailsIf<ObjectEntityDetails>()};
  return object && object->CanBeDeferredShape() &&
      evaluate::IsAllocatableOrPointer(ultimate);
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
  return derived && !IsIsoCType(derived) &&
      !derived->typeSymbol().attrs().test(Attr::BIND_C) &&
      !derived->typeSymbol().get<DerivedTypeDetails>().sequence();
}

bool IsBuiltinDerivedType(const DerivedTypeSpec *derived, const char *name) {
  if (!derived) {
    return false;
  } else {
    const auto &symbol{derived->typeSymbol()};
    return &symbol.owner() == symbol.owner().context().GetBuiltinsScope() &&
        symbol.name() == "__builtin_"s + name;
  }
}

bool IsIsoCType(const DerivedTypeSpec *derived) {
  return IsBuiltinDerivedType(derived, "c_ptr") ||
      IsBuiltinDerivedType(derived, "c_funptr");
}

bool IsTeamType(const DerivedTypeSpec *derived) {
  return IsBuiltinDerivedType(derived, "team_type");
}

bool IsBadCoarrayType(const DerivedTypeSpec *derived) {
  return IsTeamType(derived) || IsIsoCType(derived);
}

bool IsEventTypeOrLockType(const DerivedTypeSpec *derivedTypeSpec) {
  return IsBuiltinDerivedType(derivedTypeSpec, "event_type") ||
      IsBuiltinDerivedType(derivedTypeSpec, "lock_type");
}

int CountLenParameters(const DerivedTypeSpec &type) {
  return std::count_if(type.parameters().begin(), type.parameters().end(),
      [](const auto &pair) { return pair.second.isLen(); });
}

int CountNonConstantLenParameters(const DerivedTypeSpec &type) {
  return std::count_if(
      type.parameters().begin(), type.parameters().end(), [](const auto &pair) {
        if (!pair.second.isLen()) {
          return false;
        } else if (const auto &expr{pair.second.GetExplicit()}) {
          return !IsConstantExpr(*expr);
        } else {
          return true;
        }
      });
}

// Are the type parameters of type1 compile-time compatible with the
// corresponding kind type parameters of type2?  Return true if all constant
// valued parameters are equal.
// Used to check assignment statements and argument passing.  See 15.5.2.4(4)
bool AreTypeParamCompatible(const semantics::DerivedTypeSpec &type1,
    const semantics::DerivedTypeSpec &type2) {
  for (const auto &[name, param1] : type1.parameters()) {
    if (semantics::MaybeIntExpr paramExpr1{param1.GetExplicit()}) {
      if (IsConstantExpr(*paramExpr1)) {
        const semantics::ParamValue *param2{type2.FindParameter(name)};
        if (param2) {
          if (semantics::MaybeIntExpr paramExpr2{param2->GetExplicit()}) {
            if (IsConstantExpr(*paramExpr2)) {
              if (ToInt64(*paramExpr1) != ToInt64(*paramExpr2)) {
                return false;
              }
            }
          }
        }
      }
    }
  }
  return true;
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
            const Symbol *iface{proc.interface().symbol()};
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

} // namespace Fortran::semantics
