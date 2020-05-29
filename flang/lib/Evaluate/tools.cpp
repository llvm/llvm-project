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

Expr<SomeType> Parenthesize(Expr<SomeType> &&expr) {
  return std::visit(
      [&](auto &&x) {
        using T = std::decay_t<decltype(x)>;
        if constexpr (common::HasMember<T, TypelessExpression> ||
            std::is_same_v<T, Expr<SomeDerived>>) {
          return expr; // no parentheses around typeless or derived type
        } else {
          return std::visit(
              [](auto &&y) {
                using T = ResultType<decltype(y)>;
                return AsGenericExpr(Parentheses<T>{std::move(y)});
              },
              std::move(x.u));
        }
      },
      std::move(expr.u));
}

std::optional<DataRef> ExtractSubstringBase(const Substring &substring) {
  return std::visit(
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
  return !symbol.attrs().test(semantics::Attr::PARAMETER);
}
auto IsVariableHelper::operator()(const Component &x) const -> Result {
  return (*this)(x.base());
}
auto IsVariableHelper::operator()(const ArrayRef &x) const -> Result {
  return (*this)(x.base());
}
auto IsVariableHelper::operator()(const Substring &x) const -> Result {
  return (*this)(x.GetBaseObject());
}
auto IsVariableHelper::operator()(const ProcedureDesignator &x) const
    -> Result {
  const Symbol *symbol{x.GetSymbol()};
  return symbol && symbol->attrs().test(semantics::Attr::POINTER);
}

// Conversions of complex component expressions to REAL.
ConvertRealOperandsResult ConvertRealOperands(
    parser::ContextualMessages &messages, Expr<SomeType> &&x,
    Expr<SomeType> &&y, int defaultRealKind) {
  return std::visit(
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
  return Package(std::visit(
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
    return {std::visit(
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
  return std::visit(
      [&](const auto &zk) {
        static constexpr int kind{ResultType<decltype(zk)>::kind};
        return AsCategoryExpr(ComplexComponent<kind>{isImaginary, zk});
      },
      z.u);
}

// Convert REAL to COMPLEX of the same kind. Preserving the real operand kind
// and then applying complex operand promotion rules allows the result to have
// the highest precision of REAL and COMPLEX operands as required by Fortran
// 2018 10.9.1.3.
Expr<SomeComplex> PromoteRealToComplex(Expr<SomeReal> &&someX) {
  return std::visit(
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
    Expr<SomeKind<RCAT>> &&iry, int defaultRealKind) {
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
  } else if constexpr (std::is_same_v<OPR<LargestReal>,
                           Multiply<LargestReal>> ||
      std::is_same_v<OPR<LargestReal>, Divide<LargestReal>>) {
    // (a,b) * x -> (a*x, b*x)
    // (a,b) / x -> (a/x, b/x)
    auto copy{iry};
    auto rr{NumericOperation<Multiply>(messages, AsGenericExpr(std::move(zr)),
        AsGenericExpr(std::move(iry)), defaultRealKind)};
    auto ri{NumericOperation<Multiply>(messages, AsGenericExpr(std::move(zi)),
        AsGenericExpr(std::move(copy)), defaultRealKind)};
    if (auto parts{common::AllPresent(std::move(rr), std::move(ri))}) {
      return Package(ConstructComplex(messages, std::get<0>(std::move(*parts)),
          std::get<1>(std::move(*parts)), defaultRealKind));
    }
  } else if constexpr (RCAT == TypeCategory::Integer &&
      std::is_same_v<OPR<LargestReal>, Power<LargestReal>>) {
    // COMPLEX**INTEGER is a special case that doesn't convert the exponent.
    static_assert(RCAT == TypeCategory::Integer);
    return Package(std::visit(
        [&](auto &&zxk) {
          using Ty = ResultType<decltype(zxk)>;
          return AsCategoryExpr(
              AsExpr(RealToIntPower<Ty>{std::move(zxk), std::move(iry)}));
        },
        std::move(zx.u)));
  } else if (defaultRealKind != 666) { // dodge unused parameter warning
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
    Expr<SomeComplex> &&zy, int defaultRealKind) {
  if constexpr (std::is_same_v<OPR<LargestReal>, Add<LargestReal>> ||
      std::is_same_v<OPR<LargestReal>, Multiply<LargestReal>>) {
    // x + (a,b) -> (a,b) + x -> (a+x, b)
    // x * (a,b) -> (a,b) * x -> (a*x, b*x)
    return MixedComplexLeft<Add, LCAT>(
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
  } else if (defaultRealKind != 666) { // dodge unused parameter warning
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
  return std::visit(
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
            return Package(std::visit(
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
  return std::visit(
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
  return std::visit(
      [](auto &&xk) { return AsCategoryExpr(LogicalNegation(std::move(xk))); },
      std::move(x.u));
}

template <typename T>
Expr<LogicalResult> PackageRelation(
    RelationalOperator opr, Expr<T> &&x, Expr<T> &&y) {
  static_assert(IsSpecificIntrinsicType<T>);
  return Expr<LogicalResult>{
      Relational<SomeType>{Relational<T>{opr, std::move(x), std::move(y)}}};
}

template <TypeCategory CAT>
Expr<LogicalResult> PromoteAndRelate(
    RelationalOperator opr, Expr<SomeKind<CAT>> &&x, Expr<SomeKind<CAT>> &&y) {
  return std::visit(
      [=](auto &&xy) {
        return PackageRelation(opr, std::move(xy[0]), std::move(xy[1]));
      },
      AsSameKindExprs(std::move(x), std::move(y)));
}

std::optional<Expr<LogicalResult>> Relate(parser::ContextualMessages &messages,
    RelationalOperator opr, Expr<SomeType> &&x, Expr<SomeType> &&y) {
  return std::visit(
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
            if (opr != RelationalOperator::EQ &&
                opr != RelationalOperator::NE) {
              messages.Say(
                  "COMPLEX data may be compared only for equality"_err_en_US);
            } else {
              auto rr{Relate(messages, opr,
                  AsGenericExpr(GetComplexPart(zx, false)),
                  AsGenericExpr(GetComplexPart(zy, false)))};
              auto ri{
                  Relate(messages, opr, AsGenericExpr(GetComplexPart(zx, true)),
                      AsGenericExpr(GetComplexPart(zy, true)))};
              if (auto parts{
                      common::AllPresent(std::move(rr), std::move(ri))}) {
                // (a,b)==(c,d) -> (a==c) .AND. (b==d)
                // (a,b)/=(c,d) -> (a/=c) .OR. (b/=d)
                LogicalOperator combine{opr == RelationalOperator::EQ
                        ? LogicalOperator::And
                        : LogicalOperator::Or};
                return Expr<LogicalResult>{
                    LogicalOperation<LogicalResult::kind>{combine,
                        std::get<0>(std::move(*parts)),
                        std::get<1>(std::move(*parts))}};
              }
            }
            return std::nullopt;
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
            return std::visit(
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
  return std::visit(
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
  return std::visit(
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
      if (type.charLength()) {
        if (const auto &len{type.charLength()->GetExplicit()}) {
          Expr<SomeInteger> lenParam{*len};
          Expr<SubscriptInteger> length{Convert<SubscriptInteger>{lenParam}};
          converted = std::visit(
              [&](auto &&x) {
                using Ty = std::decay_t<decltype(x)>;
                using CharacterType = typename Ty::Result;
                return Expr<SomeCharacter>{
                    Expr<CharacterType>{SetLength<CharacterType::kind>{
                        std::move(x), std::move(length)}}};
              },
              std::move(converted.u));
        }
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
      if (type == *fromType) {
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
  if (int xRank{x.Rank()}; xRank > 0) {
    if (symbol.Rank() != xRank) {
      return std::nullopt;
    }
  }
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

bool IsAssumedRank(const Symbol &symbol0) {
  const Symbol &symbol{ResolveAssociations(symbol0)};
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

bool IsProcedure(const Expr<SomeType> &expr) {
  return std::holds_alternative<ProcedureDesignator>(expr.u);
}

bool IsProcedurePointer(const Expr<SomeType> &expr) {
  return std::visit(common::visitors{
                        [](const NullPointer &) { return true; },
                        [](const ProcedureDesignator &) { return true; },
                        [](const ProcedureRef &) { return true; },
                        [](const auto &) { return false; },
                    },
      expr.u);
}

// IsNullPointer()
struct IsNullPointerHelper : public AllTraverse<IsNullPointerHelper, false> {
  using Base = AllTraverse<IsNullPointerHelper, false>;
  IsNullPointerHelper() : Base(*this) {}
  using Base::operator();
  bool operator()(const ProcedureRef &call) const {
    auto *intrinsic{call.proc().GetSpecificIntrinsic()};
    return intrinsic &&
        intrinsic->characteristics.value().attrs.test(
            characteristics::Procedure::Attr::NullPointer);
  }
  bool operator()(const NullPointer &) const { return true; }
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

const Symbol &ResolveAssociations(const Symbol &symbol) {
  if (const auto *details{symbol.detailsIf<semantics::AssocEntityDetails>()}) {
    if (const Symbol * nested{UnwrapWholeSymbolDataRef(details->expr())}) {
      return ResolveAssociations(*nested);
    }
  }
  return symbol.GetUltimate();
}

struct CollectSymbolsHelper
    : public SetTraverse<CollectSymbolsHelper, semantics::SymbolSet> {
  using Base = SetTraverse<CollectSymbolsHelper, semantics::SymbolSet>;
  CollectSymbolsHelper() : Base{*this} {}
  using Base::operator();
  semantics::SymbolSet operator()(const Symbol &symbol) const {
    return {symbol};
  }
};
template <typename A> semantics::SymbolSet CollectSymbols(const A &x) {
  return CollectSymbolsHelper{}(x);
}
template semantics::SymbolSet CollectSymbols(const Expr<SomeType> &);
template semantics::SymbolSet CollectSymbols(const Expr<SomeInteger> &);
template semantics::SymbolSet CollectSymbols(const Expr<SubscriptInteger> &);

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
          "Procedure '%s' is bound to '%s'"_en_US, symbol.name(),
          binding->symbol().name());
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
  if (message) {
    AttachDeclaration(*message, symbol);
  }
  return message;
}

class FindImpureCallHelper
    : public AnyTraverse<FindImpureCallHelper, std::optional<std::string>> {
  using Result = std::optional<std::string>;
  using Base = AnyTraverse<FindImpureCallHelper, Result>;

public:
  explicit FindImpureCallHelper(const IntrinsicProcTable &intrinsics)
      : Base{*this}, intrinsics_{intrinsics} {}
  using Base::operator();
  Result operator()(const ProcedureRef &call) const {
    if (auto chars{characteristics::Procedure::Characterize(
            call.proc(), intrinsics_)}) {
      if (chars->attrs.test(characteristics::Procedure::Attr::Pure)) {
        return (*this)(call.arguments());
      }
    }
    return call.proc().GetName();
  }

private:
  const IntrinsicProcTable &intrinsics_;
};

std::optional<std::string> FindImpureCall(
    const IntrinsicProcTable &intrinsics, const Expr<SomeType> &expr) {
  return FindImpureCallHelper{intrinsics}(expr);
}
std::optional<std::string> FindImpureCall(
    const IntrinsicProcTable &intrinsics, const ProcedureRef &proc) {
  return FindImpureCallHelper{intrinsics}(proc);
}

} // namespace Fortran::evaluate

namespace Fortran::semantics {

// When a construct association maps to a variable, and that variable
// is not an array with a vector-valued subscript, return the base
// Symbol of that variable, else nullptr.  Descends into other construct
// associations when one associations maps to another.
static const Symbol *GetAssociatedVariable(
    const semantics::AssocEntityDetails &details) {
  if (const auto &expr{details.expr()}) {
    if (IsVariable(*expr) && !HasVectorSubscript(*expr)) {
      if (const Symbol * varSymbol{GetFirstSymbol(*expr)}) {
        return GetAssociationRoot(*varSymbol);
      }
    }
  }
  return nullptr;
}

const Symbol *GetAssociationRoot(const Symbol &symbol) {
  const Symbol &ultimate{symbol.GetUltimate()};
  const auto *details{ultimate.detailsIf<semantics::AssocEntityDetails>()};
  return details ? GetAssociatedVariable(*details) : &ultimate;
}

bool IsVariableName(const Symbol &symbol) {
  const Symbol *root{GetAssociationRoot(symbol)};
  return root && root->has<ObjectEntityDetails>() && !IsNamedConstant(*root);
}

bool IsPureProcedure(const Symbol &symbol) {
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
        const Symbol *root{GetAssociationRoot(*ref)};
        if (root && root->attrs().test(Attr::VOLATILE)) {
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
  return std::visit(
      common::visitors{
          [](const SubprogramDetails &x) { return x.isFunction(); },
          [&](const SubprogramNameDetails &) {
            return symbol.test(Symbol::Flag::Function);
          },
          [](const ProcEntityDetails &x) {
            const auto &ifc{x.interface()};
            return ifc.type() || (ifc.symbol() && IsFunction(*ifc.symbol()));
          },
          [](const ProcBindingDetails &x) { return IsFunction(x.symbol()); },
          [](const UseDetails &x) { return IsFunction(x.symbol()); },
          [](const auto &) { return false; },
      },
      symbol.details());
}

bool IsProcedure(const Symbol &symbol) {
  return std::visit(
      common::visitors{
          [](const SubprogramDetails &) { return true; },
          [](const SubprogramNameDetails &) { return true; },
          [](const ProcEntityDetails &) { return true; },
          [](const GenericDetails &) { return true; },
          [](const ProcBindingDetails &) { return true; },
          [](const UseDetails &x) { return IsProcedure(x.symbol()); },
          // TODO: FinalProcDetails?
          [](const auto &) { return false; },
      },
      symbol.details());
}

const Symbol *FindCommonBlockContaining(const Symbol &object) {
  const auto *details{object.detailsIf<ObjectEntityDetails>()};
  return details ? details->commonBlock() : nullptr;
}

bool IsProcedurePointer(const Symbol &symbol) {
  return symbol.has<ProcEntityDetails>() && IsPointer(symbol);
}

bool IsSaved(const Symbol &symbol) {
  auto scopeKind{symbol.owner().kind()};
  if (scopeKind == Scope::Kind::Module || scopeKind == Scope::Kind::BlockData) {
    return true;
  } else if (scopeKind == Scope::Kind::DerivedType) {
    return false; // this is a component
  } else if (IsNamedConstant(symbol)) {
    return false;
  } else if (symbol.attrs().test(Attr::SAVE)) {
    return true;
  } else if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()};
             object && object->init()) {
    return true;
  } else if (IsProcedurePointer(symbol) &&
      symbol.get<ProcEntityDetails>().init()) {
    return true;
  } else if (const Symbol * block{FindCommonBlockContaining(symbol)};
             block && block->attrs().test(Attr::SAVE)) {
    return true;
  } else {
    return false;
  }
}

bool IsDummy(const Symbol &symbol) {
  return std::visit(
      common::visitors{[](const EntityDetails &x) { return x.isDummy(); },
          [](const ObjectEntityDetails &x) { return x.isDummy(); },
          [](const ProcEntityDetails &x) { return x.isDummy(); },
          [](const HostAssocDetails &x) { return IsDummy(x.symbol()); },
          [](const auto &) { return false; }},
      symbol.details());
}

int CountLenParameters(const DerivedTypeSpec &type) {
  return std::count_if(type.parameters().begin(), type.parameters().end(),
      [](const auto &pair) { return pair.second.isLen(); });
}

const Symbol &GetUsedModule(const UseDetails &details) {
  return DEREF(details.symbol().owner().symbol());
}

} // namespace Fortran::semantics
