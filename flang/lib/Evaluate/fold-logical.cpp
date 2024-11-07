//===-- lib/Evaluate/fold-logical.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "fold-implementation.h"
#include "fold-matmul.h"
#include "fold-reduction.h"
#include "flang/Evaluate/check-expression.h"
#include "flang/Runtime/magic-numbers.h"

namespace Fortran::evaluate {

template <typename T>
static std::optional<Expr<SomeType>> ZeroExtend(const Constant<T> &c) {
  std::vector<Scalar<LargestInt>> exts;
  for (const auto &v : c.values()) {
    exts.push_back(Scalar<LargestInt>::ConvertUnsigned(v).value);
  }
  return AsGenericExpr(
      Constant<LargestInt>(std::move(exts), ConstantSubscripts(c.shape())));
}

// for ALL, ANY & PARITY
template <typename T>
static Expr<T> FoldAllAnyParity(FoldingContext &context, FunctionRef<T> &&ref,
    Scalar<T> (Scalar<T>::*operation)(const Scalar<T> &) const,
    Scalar<T> identity) {
  static_assert(T::category == TypeCategory::Logical);
  std::optional<int> dim;
  if (std::optional<ArrayAndMask<T>> arrayAndMask{
          ProcessReductionArgs<T>(context, ref.arguments(), dim,
              /*ARRAY(MASK)=*/0, /*DIM=*/1)}) {
    OperationAccumulator accumulator{arrayAndMask->array, operation};
    return Expr<T>{DoReduction<T>(
        arrayAndMask->array, arrayAndMask->mask, dim, identity, accumulator)};
  }
  return Expr<T>{std::move(ref)};
}

// OUT_OF_RANGE(x,mold[,round]) references are entirely rewritten here into
// expressions, which are then folded into constants when 'x' and 'round'
// are constant.  It is guaranteed that 'x' is evaluated at most once.

template <int X_RKIND, int MOLD_IKIND>
Expr<SomeReal> RealToIntBoundHelper(bool round, bool negate) {
  using RType = Type<TypeCategory::Real, X_RKIND>;
  using RealType = Scalar<RType>;
  using IntType = Scalar<Type<TypeCategory::Integer, MOLD_IKIND>>;
  RealType result{}; // 0.
  common::RoundingMode roundingMode{round
          ? common::RoundingMode::TiesAwayFromZero
          : common::RoundingMode::ToZero};
  // Add decreasing powers of two to the result to find the largest magnitude
  // value that can be converted to the integer type without overflow.
  RealType at{RealType::FromInteger(IntType{negate ? -1 : 1}).value};
  bool decrement{true};
  while (!at.template ToInteger<IntType>(roundingMode)
              .flags.test(RealFlag::Overflow)) {
    auto tmp{at.SCALE(IntType{1})};
    if (tmp.flags.test(RealFlag::Overflow)) {
      decrement = false;
      break;
    }
    at = tmp.value;
  }
  while (true) {
    if (decrement) {
      at = at.SCALE(IntType{-1}).value;
    } else {
      decrement = true;
    }
    auto tmp{at.Add(result)};
    if (tmp.flags.test(RealFlag::Inexact)) {
      break;
    } else if (!tmp.value.template ToInteger<IntType>(roundingMode)
                    .flags.test(RealFlag::Overflow)) {
      result = tmp.value;
    }
  }
  return AsCategoryExpr(Constant<RType>{std::move(result)});
}

static Expr<SomeReal> RealToIntBound(
    int xRKind, int moldIKind, bool round, bool negate) {
  switch (xRKind) {
#define ICASES(RK) \
  switch (moldIKind) { \
  case 1: \
    return RealToIntBoundHelper<RK, 1>(round, negate); \
    break; \
  case 2: \
    return RealToIntBoundHelper<RK, 2>(round, negate); \
    break; \
  case 4: \
    return RealToIntBoundHelper<RK, 4>(round, negate); \
    break; \
  case 8: \
    return RealToIntBoundHelper<RK, 8>(round, negate); \
    break; \
  case 16: \
    return RealToIntBoundHelper<RK, 16>(round, negate); \
    break; \
  } \
  break
  case 2:
    ICASES(2);
    break;
  case 3:
    ICASES(3);
    break;
  case 4:
    ICASES(4);
    break;
  case 8:
    ICASES(8);
    break;
  case 10:
    ICASES(10);
    break;
  case 16:
    ICASES(16);
    break;
  }
  DIE("RealToIntBound: no case");
#undef ICASES
}

class RealToIntLimitHelper {
public:
  using Result = std::optional<Expr<SomeReal>>;
  using Types = RealTypes;
  RealToIntLimitHelper(
      FoldingContext &context, Expr<SomeReal> &&hi, Expr<SomeReal> &lo)
      : context_{context}, hi_{std::move(hi)}, lo_{lo} {}
  template <typename T> Result Test() {
    if (UnwrapExpr<Expr<T>>(hi_)) {
      bool promote{T::kind < 16};
      Result constResult;
      if (auto hiV{GetScalarConstantValue<T>(hi_)}) {
        auto loV{GetScalarConstantValue<T>(lo_)};
        CHECK(loV.has_value());
        auto diff{hiV->Subtract(*loV, Rounding{common::RoundingMode::ToZero})};
        promote = promote &&
            (diff.flags.test(RealFlag::Overflow) ||
                diff.flags.test(RealFlag::Inexact));
        constResult = AsCategoryExpr(Constant<T>{std::move(diff.value)});
      }
      if (promote) {
        constexpr int nextKind{T::kind < 4 ? 4 : T::kind == 4 ? 8 : 16};
        using T2 = Type<TypeCategory::Real, nextKind>;
        hi_ = Expr<SomeReal>{Fold(context_, ConvertToType<T2>(std::move(hi_)))};
        lo_ = Expr<SomeReal>{Fold(context_, ConvertToType<T2>(std::move(lo_)))};
        if (constResult) {
          // Use promoted constants on next iteration of SearchTypes
          return std::nullopt;
        }
      }
      if (constResult) {
        return constResult;
      } else {
        return AsCategoryExpr(std::move(hi_) - Expr<SomeReal>{lo_});
      }
    } else {
      return std::nullopt;
    }
  }

private:
  FoldingContext &context_;
  Expr<SomeReal> hi_;
  Expr<SomeReal> &lo_;
};

static std::optional<Expr<SomeReal>> RealToIntLimit(
    FoldingContext &context, Expr<SomeReal> &&hi, Expr<SomeReal> &lo) {
  return common::SearchTypes(RealToIntLimitHelper{context, std::move(hi), lo});
}

// RealToRealBounds() returns a pair (HUGE(x),REAL(HUGE(mold),KIND(x)))
// when REAL(HUGE(x),KIND(mold)) overflows, and std::nullopt otherwise.
template <int X_RKIND, int MOLD_RKIND>
std::optional<std::pair<Expr<SomeReal>, Expr<SomeReal>>>
RealToRealBoundsHelper() {
  using RType = Type<TypeCategory::Real, X_RKIND>;
  using RealType = Scalar<RType>;
  using MoldRealType = Scalar<Type<TypeCategory::Real, MOLD_RKIND>>;
  if (!MoldRealType::Convert(RealType::HUGE()).flags.test(RealFlag::Overflow)) {
    return std::nullopt;
  } else {
    return std::make_pair(AsCategoryExpr(Constant<RType>{
                              RealType::Convert(MoldRealType::HUGE()).value}),
        AsCategoryExpr(Constant<RType>{RealType::HUGE()}));
  }
}

static std::optional<std::pair<Expr<SomeReal>, Expr<SomeReal>>>
RealToRealBounds(int xRKind, int moldRKind) {
  switch (xRKind) {
#define RCASES(RK) \
  switch (moldRKind) { \
  case 2: \
    return RealToRealBoundsHelper<RK, 2>(); \
    break; \
  case 3: \
    return RealToRealBoundsHelper<RK, 3>(); \
    break; \
  case 4: \
    return RealToRealBoundsHelper<RK, 4>(); \
    break; \
  case 8: \
    return RealToRealBoundsHelper<RK, 8>(); \
    break; \
  case 10: \
    return RealToRealBoundsHelper<RK, 10>(); \
    break; \
  case 16: \
    return RealToRealBoundsHelper<RK, 16>(); \
    break; \
  } \
  break
  case 2:
    RCASES(2);
    break;
  case 3:
    RCASES(3);
    break;
  case 4:
    RCASES(4);
    break;
  case 8:
    RCASES(8);
    break;
  case 10:
    RCASES(10);
    break;
  case 16:
    RCASES(16);
    break;
  }
  DIE("RealToRealBounds: no case");
#undef RCASES
}

template <int X_IKIND, int MOLD_RKIND>
std::optional<Expr<SomeInteger>> IntToRealBoundHelper(bool negate) {
  using IType = Type<TypeCategory::Integer, X_IKIND>;
  using IntType = Scalar<IType>;
  using RealType = Scalar<Type<TypeCategory::Real, MOLD_RKIND>>;
  IntType result{}; // 0
  while (true) {
    std::optional<IntType> next;
    for (int bit{0}; bit < IntType::bits; ++bit) {
      IntType power{IntType{}.IBSET(bit)};
      if (power.IsNegative()) {
        if (!negate) {
          break;
        }
      } else if (negate) {
        power = power.Negate().value;
      }
      auto tmp{power.AddSigned(result)};
      if (tmp.overflow ||
          RealType::FromInteger(tmp.value).flags.test(RealFlag::Overflow)) {
        break;
      }
      next = tmp.value;
    }
    if (next) {
      CHECK(result.CompareSigned(*next) != Ordering::Equal);
      result = *next;
    } else {
      break;
    }
  }
  if (result.CompareSigned(IntType::HUGE()) == Ordering::Equal) {
    return std::nullopt;
  } else {
    return AsCategoryExpr(Constant<IType>{std::move(result)});
  }
}

static std::optional<Expr<SomeInteger>> IntToRealBound(
    int xIKind, int moldRKind, bool negate) {
  switch (xIKind) {
#define RCASES(IK) \
  switch (moldRKind) { \
  case 2: \
    return IntToRealBoundHelper<IK, 2>(negate); \
    break; \
  case 3: \
    return IntToRealBoundHelper<IK, 3>(negate); \
    break; \
  case 4: \
    return IntToRealBoundHelper<IK, 4>(negate); \
    break; \
  case 8: \
    return IntToRealBoundHelper<IK, 8>(negate); \
    break; \
  case 10: \
    return IntToRealBoundHelper<IK, 10>(negate); \
    break; \
  case 16: \
    return IntToRealBoundHelper<IK, 16>(negate); \
    break; \
  } \
  break
  case 1:
    RCASES(1);
    break;
  case 2:
    RCASES(2);
    break;
  case 4:
    RCASES(4);
    break;
  case 8:
    RCASES(8);
    break;
  case 16:
    RCASES(16);
    break;
  }
  DIE("IntToRealBound: no case");
#undef RCASES
}

template <int X_IKIND, int MOLD_IKIND>
std::optional<Expr<SomeInteger>> IntToIntBoundHelper() {
  if constexpr (X_IKIND <= MOLD_IKIND) {
    return std::nullopt;
  } else {
    using XIType = Type<TypeCategory::Integer, X_IKIND>;
    using IntegerType = Scalar<XIType>;
    using MoldIType = Type<TypeCategory::Integer, MOLD_IKIND>;
    using MoldIntegerType = Scalar<MoldIType>;
    return AsCategoryExpr(Constant<XIType>{
        IntegerType::ConvertSigned(MoldIntegerType::HUGE()).value});
  }
}

static std::optional<Expr<SomeInteger>> IntToIntBound(
    int xIKind, int moldIKind) {
  switch (xIKind) {
#define ICASES(IK) \
  switch (moldIKind) { \
  case 1: \
    return IntToIntBoundHelper<IK, 1>(); \
    break; \
  case 2: \
    return IntToIntBoundHelper<IK, 2>(); \
    break; \
  case 4: \
    return IntToIntBoundHelper<IK, 4>(); \
    break; \
  case 8: \
    return IntToIntBoundHelper<IK, 8>(); \
    break; \
  case 16: \
    return IntToIntBoundHelper<IK, 16>(); \
    break; \
  } \
  break
  case 1:
    ICASES(1);
    break;
  case 2:
    ICASES(2);
    break;
  case 4:
    ICASES(4);
    break;
  case 8:
    ICASES(8);
    break;
  case 16:
    ICASES(16);
    break;
  }
  DIE("IntToIntBound: no case");
#undef ICASES
}

// ApplyIntrinsic() constructs the typed expression representation
// for a specific intrinsic function reference.
// TODO: maybe move into tools.h?
class IntrinsicCallHelper {
public:
  explicit IntrinsicCallHelper(SpecificCall &&call) : call_{call} {
    CHECK(proc_.IsFunction());
    typeAndShape_ = proc_.functionResult->GetTypeAndShape();
    CHECK(typeAndShape_ != nullptr);
  }
  using Result = std::optional<Expr<SomeType>>;
  using Types = LengthlessIntrinsicTypes;
  template <typename T> Result Test() {
    if (T::category == typeAndShape_->type().category() &&
        T::kind == typeAndShape_->type().kind()) {
      return AsGenericExpr(FunctionRef<T>{
          ProcedureDesignator{std::move(call_.specificIntrinsic)},
          std::move(call_.arguments)});
    } else {
      return std::nullopt;
    }
  }

private:
  SpecificCall call_;
  const characteristics::Procedure &proc_{
      call_.specificIntrinsic.characteristics.value()};
  const characteristics::TypeAndShape *typeAndShape_{nullptr};
};

static Expr<SomeType> ApplyIntrinsic(
    FoldingContext &context, const std::string &func, ActualArguments &&args) {
  auto found{
      context.intrinsics().Probe(CallCharacteristics{func}, args, context)};
  CHECK(found.has_value());
  auto result{common::SearchTypes(IntrinsicCallHelper{std::move(*found)})};
  CHECK(result.has_value());
  return *result;
}

static Expr<LogicalResult> CompareUnsigned(FoldingContext &context,
    const char *intrin, Expr<SomeType> &&x, Expr<SomeType> &&y) {
  Expr<SomeType> result{ApplyIntrinsic(context, intrin,
      ActualArguments{
          ActualArgument{std::move(x)}, ActualArgument{std::move(y)}})};
  return DEREF(UnwrapExpr<Expr<LogicalResult>>(result));
}

// Determines the right kind of INTEGER to hold the bits of a REAL type.
static Expr<SomeType> IntTransferMold(
    const TargetCharacteristics &target, DynamicType realType, bool asVector) {
  CHECK(realType.category() == TypeCategory::Real);
  int rKind{realType.kind()};
  int iKind{std::max<int>(target.GetAlignment(TypeCategory::Real, rKind),
      target.GetByteSize(TypeCategory::Real, rKind))};
  CHECK(target.CanSupportType(TypeCategory::Integer, iKind));
  DynamicType iType{TypeCategory::Integer, iKind};
  ConstantSubscripts shape;
  if (asVector) {
    shape = ConstantSubscripts{1};
  }
  Constant<SubscriptInteger> value{
      std::vector<Scalar<SubscriptInteger>>{0}, std::move(shape)};
  auto expr{ConvertToType(iType, AsGenericExpr(std::move(value)))};
  CHECK(expr.has_value());
  return std::move(*expr);
}

static Expr<SomeType> GetRealBits(FoldingContext &context, Expr<SomeReal> &&x) {
  auto xType{x.GetType()};
  CHECK(xType.has_value());
  bool asVector{x.Rank() > 0};
  return ApplyIntrinsic(context, "transfer",
      ActualArguments{ActualArgument{AsGenericExpr(std::move(x))},
          ActualArgument{IntTransferMold(
              context.targetCharacteristics(), *xType, asVector)}});
}

template <int KIND>
static Expr<Type<TypeCategory::Logical, KIND>> RewriteOutOfRange(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Logical, KIND>> &&funcRef) {
  using ResultType = Type<TypeCategory::Logical, KIND>;
  ActualArguments &args{funcRef.arguments()};
  // Fold x= and round= unconditionally
  if (auto *x{UnwrapExpr<Expr<SomeType>>(args[0])}) {
    *args[0] = Fold(context, std::move(*x));
  }
  if (args.size() >= 3) {
    if (auto *round{UnwrapExpr<Expr<SomeType>>(args[2])}) {
      *args[2] = Fold(context, std::move(*round));
    }
  }
  if (auto *x{UnwrapExpr<Expr<SomeType>>(args[0])}) {
    x = UnwrapExpr<Expr<SomeType>>(args[0]);
    CHECK(x != nullptr);
    if (const auto *mold{UnwrapExpr<Expr<SomeType>>(args[1])}) {
      DynamicType xType{x->GetType().value()};
      std::optional<Expr<LogicalResult>> result;
      bool alwaysFalse{false};
      if (auto *iXExpr{UnwrapExpr<Expr<SomeInteger>>(*x)}) {
        int iXKind{iXExpr->GetType().value().kind()};
        if (auto *iMoldExpr{UnwrapExpr<Expr<SomeInteger>>(*mold)}) {
          // INTEGER -> INTEGER
          int iMoldKind{iMoldExpr->GetType().value().kind()};
          if (auto hi{IntToIntBound(iXKind, iMoldKind)}) {
            // 'hi' is INT(HUGE(mold), KIND(x))
            // OUT_OF_RANGE(x,mold) = (x + (hi + 1)) .UGT. (2*hi + 1)
            auto one{DEREF(UnwrapExpr<Expr<SomeInteger>>(ConvertToType(
                xType, AsGenericExpr(Constant<SubscriptInteger>{1}))))};
            auto lhs{std::move(*iXExpr) +
                (Expr<SomeInteger>{*hi} + Expr<SomeInteger>{one})};
            auto two{DEREF(UnwrapExpr<Expr<SomeInteger>>(ConvertToType(
                xType, AsGenericExpr(Constant<SubscriptInteger>{2}))))};
            auto rhs{std::move(two) * std::move(*hi) + std::move(one)};
            result = CompareUnsigned(context, "bgt",
                Expr<SomeType>{std::move(lhs)}, Expr<SomeType>{std::move(rhs)});
          } else {
            alwaysFalse = true;
          }
        } else if (auto *rMoldExpr{UnwrapExpr<Expr<SomeReal>>(*mold)}) {
          // INTEGER -> REAL
          int rMoldKind{rMoldExpr->GetType().value().kind()};
          if (auto hi{IntToRealBound(iXKind, rMoldKind, /*negate=*/false)}) {
            // OUT_OF_RANGE(x,mold) = (x - lo) .UGT. (hi - lo)
            auto lo{IntToRealBound(iXKind, rMoldKind, /*negate=*/true)};
            CHECK(lo.has_value());
            auto lhs{std::move(*iXExpr) - Expr<SomeInteger>{*lo}};
            auto rhs{std::move(*hi) - std::move(*lo)};
            result = CompareUnsigned(context, "bgt",
                Expr<SomeType>{std::move(lhs)}, Expr<SomeType>{std::move(rhs)});
          } else {
            alwaysFalse = true;
          }
        }
      } else if (auto *rXExpr{UnwrapExpr<Expr<SomeReal>>(*x)}) {
        int rXKind{rXExpr->GetType().value().kind()};
        if (auto *iMoldExpr{UnwrapExpr<Expr<SomeInteger>>(*mold)}) {
          // REAL -> INTEGER
          int iMoldKind{iMoldExpr->GetType().value().kind()};
          auto hi{RealToIntBound(rXKind, iMoldKind, false, false)};
          auto lo{RealToIntBound(rXKind, iMoldKind, false, true)};
          if (args.size() >= 3) {
            // Bounds depend on round= value
            if (auto *round{UnwrapExpr<Expr<SomeType>>(args[2])}) {
              if (const Symbol * whole{UnwrapWholeSymbolDataRef(*round)};
                  whole && semantics::IsOptional(whole->GetUltimate()) &&
                  context.languageFeatures().ShouldWarn(
                      common::UsageWarning::OptionalMustBePresent)) {
                if (auto source{args[2]->sourceLocation()}) {
                  context.messages().Say(
                      common::UsageWarning::OptionalMustBePresent, *source,
                      "ROUND= argument to OUT_OF_RANGE() is an optional dummy argument that must be present at execution"_warn_en_US);
                }
              }
              auto rlo{RealToIntBound(rXKind, iMoldKind, true, true)};
              auto rhi{RealToIntBound(rXKind, iMoldKind, true, false)};
              auto mlo{Fold(context,
                  ApplyIntrinsic(context, "merge",
                      ActualArguments{
                          ActualArgument{Expr<SomeType>{std::move(rlo)}},
                          ActualArgument{Expr<SomeType>{std::move(lo)}},
                          ActualArgument{Expr<SomeType>{*round}}}))};
              auto mhi{Fold(context,
                  ApplyIntrinsic(context, "merge",
                      ActualArguments{
                          ActualArgument{Expr<SomeType>{std::move(rhi)}},
                          ActualArgument{Expr<SomeType>{std::move(hi)}},
                          ActualArgument{std::move(*round)}}))};
              lo = std::move(DEREF(UnwrapExpr<Expr<SomeReal>>(mlo)));
              hi = std::move(DEREF(UnwrapExpr<Expr<SomeReal>>(mhi)));
            }
          }
          // OUT_OF_RANGE(x,mold[,round]) =
          //   TRANSFER(x - lo, int) .UGT. TRANSFER(hi - lo, int)
          hi = Fold(context, std::move(hi));
          lo = Fold(context, std::move(lo));
          if (auto rhs{RealToIntLimit(context, std::move(hi), lo)}) {
            Expr<SomeReal> lhs{std::move(*rXExpr) - std::move(lo)};
            result = CompareUnsigned(context, "bgt",
                GetRealBits(context, std::move(lhs)),
                GetRealBits(context, std::move(*rhs)));
          }
        } else if (auto *rMoldExpr{UnwrapExpr<Expr<SomeReal>>(*mold)}) {
          // REAL -> REAL
          // Only finite arguments with ABS(x) > HUGE(mold) are .TRUE.
          // OUT_OF_RANGE(x,mold) =
          //   TRANSFER(ABS(x) - HUGE(mold), int) - 1 .ULT.
          //   TRANSFER(HUGE(mold), int)
          // Note that OUT_OF_RANGE(+/-Inf or NaN,mold) =
          //   TRANSFER(+Inf or Nan, int) - 1 .ULT. TRANSFER(HUGE(mold), int)
          int rMoldKind{rMoldExpr->GetType().value().kind()};
          if (auto bounds{RealToRealBounds(rXKind, rMoldKind)}) {
            auto &[moldHuge, xHuge]{*bounds};
            Expr<SomeType> abs{ApplyIntrinsic(context, "abs",
                ActualArguments{
                    ActualArgument{Expr<SomeType>{std::move(*rXExpr)}}})};
            auto &absR{DEREF(UnwrapExpr<Expr<SomeReal>>(abs))};
            Expr<SomeType> diffBits{
                GetRealBits(context, std::move(absR) - std::move(moldHuge))};
            auto &diffBitsI{DEREF(UnwrapExpr<Expr<SomeInteger>>(diffBits))};
            Expr<SomeType> decr{std::move(diffBitsI) -
                Expr<SomeInteger>{Expr<SubscriptInteger>{1}}};
            result = CompareUnsigned(context, "blt", std::move(decr),
                GetRealBits(context, std::move(xHuge)));
          } else {
            alwaysFalse = true;
          }
        }
      }
      if (alwaysFalse) {
        // xType can never overflow moldType, so
        //   OUT_OF_RANGE(x) = (x /= 0) .AND. .FALSE.
        // which has the same shape as x.
        Expr<LogicalResult> scalarFalse{
            Constant<LogicalResult>{Scalar<LogicalResult>{false}}};
        if (x->Rank() > 0) {
          if (auto nez{Relate(context.messages(), RelationalOperator::NE,
                  std::move(*x),
                  AsGenericExpr(Constant<SubscriptInteger>{0}))}) {
            result = Expr<LogicalResult>{LogicalOperation<LogicalResult::kind>{
                LogicalOperator::And, std::move(*nez), std::move(scalarFalse)}};
          }
        } else {
          result = std::move(scalarFalse);
        }
      }
      if (result) {
        auto restorer{context.messages().DiscardMessages()};
        return Fold(
            context, AsExpr(ConvertToType<ResultType>(std::move(*result))));
      }
    }
  }
  return AsExpr(std::move(funcRef));
}

static std::optional<common::RoundingMode> GetRoundingMode(
    const std::optional<ActualArgument> &arg) {
  if (arg) {
    if (const auto *cst{UnwrapExpr<Constant<SomeDerived>>(*arg)}) {
      if (auto constr{cst->GetScalarValue()}) {
        if (StructureConstructorValues & values{constr->values()};
            values.size() == 1) {
          const Expr<SomeType> &value{values.begin()->second.value()};
          if (auto code{ToInt64(value)}) {
            return static_cast<common::RoundingMode>(*code);
          }
        }
      }
    }
  }
  return std::nullopt;
}

template <int KIND>
Expr<Type<TypeCategory::Logical, KIND>> FoldIntrinsicFunction(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Logical, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Logical, KIND>;
  ActualArguments &args{funcRef.arguments()};
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  using SameInt = Type<TypeCategory::Integer, KIND>;
  if (name == "all") {
    return FoldAllAnyParity(
        context, std::move(funcRef), &Scalar<T>::AND, Scalar<T>{true});
  } else if (name == "any") {
    return FoldAllAnyParity(
        context, std::move(funcRef), &Scalar<T>::OR, Scalar<T>{false});
  } else if (name == "associated") {
    bool gotConstant{true};
    const Expr<SomeType> *firstArgExpr{args[0]->UnwrapExpr()};
    if (!firstArgExpr || !IsNullPointer(*firstArgExpr)) {
      gotConstant = false;
    } else if (args[1]) { // There's a second argument
      const Expr<SomeType> *secondArgExpr{args[1]->UnwrapExpr()};
      if (!secondArgExpr || !IsNullPointer(*secondArgExpr)) {
        gotConstant = false;
      }
    }
    return gotConstant ? Expr<T>{false} : Expr<T>{std::move(funcRef)};
  } else if (name == "bge" || name == "bgt" || name == "ble" || name == "blt") {
    static_assert(std::is_same_v<Scalar<LargestInt>, BOZLiteralConstant>);

    // The arguments to these intrinsics can be of different types. In that
    // case, the shorter of the two would need to be zero-extended to match
    // the size of the other. If at least one of the operands is not a constant,
    // the zero-extending will be done during lowering. Otherwise, the folding
    // must be done here.
    std::optional<Expr<SomeType>> constArgs[2];
    for (int i{0}; i <= 1; i++) {
      if (BOZLiteralConstant * x{UnwrapExpr<BOZLiteralConstant>(args[i])}) {
        constArgs[i] = AsGenericExpr(Constant<LargestInt>{std::move(*x)});
      } else if (auto *x{UnwrapExpr<Expr<SomeInteger>>(args[i])}) {
        common::visit(
            [&](const auto &ix) {
              using IntT = typename std::decay_t<decltype(ix)>::Result;
              if (auto *c{UnwrapConstantValue<IntT>(ix)}) {
                constArgs[i] = ZeroExtend(*c);
              }
            },
            x->u);
      }
    }

    if (constArgs[0] && constArgs[1]) {
      auto fptr{&Scalar<LargestInt>::BGE};
      if (name == "bge") { // done in fptr declaration
      } else if (name == "bgt") {
        fptr = &Scalar<LargestInt>::BGT;
      } else if (name == "ble") {
        fptr = &Scalar<LargestInt>::BLE;
      } else if (name == "blt") {
        fptr = &Scalar<LargestInt>::BLT;
      } else {
        common::die("missing case to fold intrinsic function %s", name.c_str());
      }

      for (int i{0}; i <= 1; i++) {
        *args[i] = std::move(constArgs[i].value());
      }

      return FoldElementalIntrinsic<T, LargestInt, LargestInt>(context,
          std::move(funcRef),
          ScalarFunc<T, LargestInt, LargestInt>(
              [&fptr](
                  const Scalar<LargestInt> &i, const Scalar<LargestInt> &j) {
                return Scalar<T>{std::invoke(fptr, i, j)};
              }));
    } else {
      return Expr<T>{std::move(funcRef)};
    }
  } else if (name == "btest") {
    if (const auto *ix{UnwrapExpr<Expr<SomeInteger>>(args[0])}) {
      return common::visit(
          [&](const auto &x) {
            using IT = ResultType<decltype(x)>;
            return FoldElementalIntrinsic<T, IT, SameInt>(context,
                std::move(funcRef),
                ScalarFunc<T, IT, SameInt>(
                    [&](const Scalar<IT> &x, const Scalar<SameInt> &pos) {
                      auto posVal{pos.ToInt64()};
                      if (posVal < 0 || posVal >= x.bits) {
                        context.messages().Say(
                            "POS=%jd out of range for BTEST"_err_en_US,
                            static_cast<std::intmax_t>(posVal));
                      }
                      return Scalar<T>{x.BTEST(posVal)};
                    }));
          },
          ix->u);
    }
  } else if (name == "dot_product") {
    return FoldDotProduct<T>(context, std::move(funcRef));
  } else if (name == "extends_type_of") {
    // Type extension testing with EXTENDS_TYPE_OF() ignores any type
    // parameters. Returns a constant truth value when the result is known now.
    if (args[0] && args[1]) {
      auto t0{args[0]->GetType()};
      auto t1{args[1]->GetType()};
      if (t0 && t1) {
        if (auto result{t0->ExtendsTypeOf(*t1)}) {
          return Expr<T>{*result};
        }
      }
    }
  } else if (name == "isnan" || name == "__builtin_ieee_is_nan") {
    // Only replace the type of the function if we can do the fold
    if (args[0] && args[0]->UnwrapExpr() &&
        IsActuallyConstant(*args[0]->UnwrapExpr())) {
      auto restorer{context.messages().DiscardMessages()};
      using DefaultReal = Type<TypeCategory::Real, 4>;
      return FoldElementalIntrinsic<T, DefaultReal>(context, std::move(funcRef),
          ScalarFunc<T, DefaultReal>([](const Scalar<DefaultReal> &x) {
            return Scalar<T>{x.IsNotANumber()};
          }));
    }
  } else if (name == "__builtin_ieee_is_negative") {
    auto restorer{context.messages().DiscardMessages()};
    using DefaultReal = Type<TypeCategory::Real, 4>;
    if (args[0] && args[0]->UnwrapExpr() &&
        IsActuallyConstant(*args[0]->UnwrapExpr())) {
      return FoldElementalIntrinsic<T, DefaultReal>(context, std::move(funcRef),
          ScalarFunc<T, DefaultReal>([](const Scalar<DefaultReal> &x) {
            return Scalar<T>{x.IsNegative()};
          }));
    }
  } else if (name == "__builtin_ieee_is_normal") {
    auto restorer{context.messages().DiscardMessages()};
    using DefaultReal = Type<TypeCategory::Real, 4>;
    if (args[0] && args[0]->UnwrapExpr() &&
        IsActuallyConstant(*args[0]->UnwrapExpr())) {
      return FoldElementalIntrinsic<T, DefaultReal>(context, std::move(funcRef),
          ScalarFunc<T, DefaultReal>([](const Scalar<DefaultReal> &x) {
            return Scalar<T>{x.IsNormal()};
          }));
    }
  } else if (name == "is_contiguous") {
    if (args.at(0)) {
      if (auto *expr{args[0]->UnwrapExpr()}) {
        if (auto contiguous{IsContiguous(*expr, context)}) {
          return Expr<T>{*contiguous};
        }
      } else if (auto *assumedType{args[0]->GetAssumedTypeDummy()}) {
        if (auto contiguous{IsContiguous(*assumedType, context)}) {
          return Expr<T>{*contiguous};
        }
      }
    }
  } else if (name == "is_iostat_end") {
    if (args[0] && args[0]->UnwrapExpr() &&
        IsActuallyConstant(*args[0]->UnwrapExpr())) {
      using Int64 = Type<TypeCategory::Integer, 8>;
      return FoldElementalIntrinsic<T, Int64>(context, std::move(funcRef),
          ScalarFunc<T, Int64>([](const Scalar<Int64> &x) {
            return Scalar<T>{x.ToInt64() == FORTRAN_RUNTIME_IOSTAT_END};
          }));
    }
  } else if (name == "is_iostat_eor") {
    if (args[0] && args[0]->UnwrapExpr() &&
        IsActuallyConstant(*args[0]->UnwrapExpr())) {
      using Int64 = Type<TypeCategory::Integer, 8>;
      return FoldElementalIntrinsic<T, Int64>(context, std::move(funcRef),
          ScalarFunc<T, Int64>([](const Scalar<Int64> &x) {
            return Scalar<T>{x.ToInt64() == FORTRAN_RUNTIME_IOSTAT_EOR};
          }));
    }
  } else if (name == "lge" || name == "lgt" || name == "lle" || name == "llt") {
    // Rewrite LGE/LGT/LLE/LLT into ASCII character relations
    auto *cx0{UnwrapExpr<Expr<SomeCharacter>>(args[0])};
    auto *cx1{UnwrapExpr<Expr<SomeCharacter>>(args[1])};
    if (cx0 && cx1) {
      return Fold(context,
          ConvertToType<T>(
              PackageRelation(name == "lge" ? RelationalOperator::GE
                      : name == "lgt"       ? RelationalOperator::GT
                      : name == "lle"       ? RelationalOperator::LE
                                            : RelationalOperator::LT,
                  ConvertToType<Ascii>(std::move(*cx0)),
                  ConvertToType<Ascii>(std::move(*cx1)))));
    }
  } else if (name == "logical") {
    if (auto *expr{UnwrapExpr<Expr<SomeLogical>>(args[0])}) {
      return Fold(context, ConvertToType<T>(std::move(*expr)));
    }
  } else if (name == "matmul") {
    return FoldMatmul(context, std::move(funcRef));
  } else if (name == "out_of_range") {
    return RewriteOutOfRange<KIND>(context, std::move(funcRef));
  } else if (name == "parity") {
    return FoldAllAnyParity(
        context, std::move(funcRef), &Scalar<T>::NEQV, Scalar<T>{false});
  } else if (name == "same_type_as") {
    // Type equality testing with SAME_TYPE_AS() ignores any type parameters.
    // Returns a constant truth value when the result is known now.
    if (args[0] && args[1]) {
      auto t0{args[0]->GetType()};
      auto t1{args[1]->GetType()};
      if (t0 && t1) {
        if (auto result{t0->SameTypeAs(*t1)}) {
          return Expr<T>{*result};
        }
      }
    }
  } else if (name == "__builtin_ieee_support_datatype") {
    return Expr<T>{true};
  } else if (name == "__builtin_ieee_support_denormal") {
    return Expr<T>{context.targetCharacteristics().ieeeFeatures().test(
        IeeeFeature::Denormal)};
  } else if (name == "__builtin_ieee_support_divide") {
    return Expr<T>{context.targetCharacteristics().ieeeFeatures().test(
        IeeeFeature::Divide)};
  } else if (name == "__builtin_ieee_support_flag") {
    return Expr<T>{context.targetCharacteristics().ieeeFeatures().test(
        IeeeFeature::Flags)};
  } else if (name == "__builtin_ieee_support_halting") {
    return Expr<T>{context.targetCharacteristics().ieeeFeatures().test(
        IeeeFeature::Halting)};
  } else if (name == "__builtin_ieee_support_inf") {
    return Expr<T>{
        context.targetCharacteristics().ieeeFeatures().test(IeeeFeature::Inf)};
  } else if (name == "__builtin_ieee_support_io") {
    return Expr<T>{
        context.targetCharacteristics().ieeeFeatures().test(IeeeFeature::Io)};
  } else if (name == "__builtin_ieee_support_nan") {
    return Expr<T>{
        context.targetCharacteristics().ieeeFeatures().test(IeeeFeature::NaN)};
  } else if (name == "__builtin_ieee_support_rounding") {
    if (context.targetCharacteristics().ieeeFeatures().test(
            IeeeFeature::Rounding)) {
      if (auto mode{GetRoundingMode(args[0])}) {
        return Expr<T>{mode != common::RoundingMode::TiesAwayFromZero};
      }
    }
  } else if (name == "__builtin_ieee_support_sqrt") {
    return Expr<T>{
        context.targetCharacteristics().ieeeFeatures().test(IeeeFeature::Sqrt)};
  } else if (name == "__builtin_ieee_support_standard") {
    return Expr<T>{context.targetCharacteristics().ieeeFeatures().test(
        IeeeFeature::Standard)};
  } else if (name == "__builtin_ieee_support_subnormal") {
    return Expr<T>{context.targetCharacteristics().ieeeFeatures().test(
        IeeeFeature::Subnormal)};
  } else if (name == "__builtin_ieee_support_underflow_control") {
    return Expr<T>{context.targetCharacteristics().ieeeFeatures().test(
        IeeeFeature::UnderflowControl)};
  }
  return Expr<T>{std::move(funcRef)};
}

template <typename T>
Expr<LogicalResult> FoldOperation(
    FoldingContext &context, Relational<T> &&relation) {
  if (auto array{ApplyElementwise(context, relation,
          std::function<Expr<LogicalResult>(Expr<T> &&, Expr<T> &&)>{
              [=](Expr<T> &&x, Expr<T> &&y) {
                return Expr<LogicalResult>{Relational<SomeType>{
                    Relational<T>{relation.opr, std::move(x), std::move(y)}}};
              }})}) {
    return *array;
  }
  if (auto folded{OperandsAreConstants(relation)}) {
    bool result{};
    if constexpr (T::category == TypeCategory::Integer) {
      result =
          Satisfies(relation.opr, folded->first.CompareSigned(folded->second));
    } else if constexpr (T::category == TypeCategory::Real) {
      result = Satisfies(relation.opr, folded->first.Compare(folded->second));
    } else if constexpr (T::category == TypeCategory::Complex) {
      result = (relation.opr == RelationalOperator::EQ) ==
          folded->first.Equals(folded->second);
    } else if constexpr (T::category == TypeCategory::Character) {
      result = Satisfies(relation.opr, Compare(folded->first, folded->second));
    } else {
      static_assert(T::category != TypeCategory::Logical);
    }
    return Expr<LogicalResult>{Constant<LogicalResult>{result}};
  }
  return Expr<LogicalResult>{Relational<SomeType>{std::move(relation)}};
}

Expr<LogicalResult> FoldOperation(
    FoldingContext &context, Relational<SomeType> &&relation) {
  return common::visit(
      [&](auto &&x) {
        return Expr<LogicalResult>{FoldOperation(context, std::move(x))};
      },
      std::move(relation.u));
}

template <int KIND>
Expr<Type<TypeCategory::Logical, KIND>> FoldOperation(
    FoldingContext &context, Not<KIND> &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  using Ty = Type<TypeCategory::Logical, KIND>;
  auto &operand{x.left()};
  if (auto value{GetScalarConstantValue<Ty>(operand)}) {
    return Expr<Ty>{Constant<Ty>{!value->IsTrue()}};
  }
  return Expr<Ty>{x};
}

template <int KIND>
Expr<Type<TypeCategory::Logical, KIND>> FoldOperation(
    FoldingContext &context, LogicalOperation<KIND> &&operation) {
  using LOGICAL = Type<TypeCategory::Logical, KIND>;
  if (auto array{ApplyElementwise(context, operation,
          std::function<Expr<LOGICAL>(Expr<LOGICAL> &&, Expr<LOGICAL> &&)>{
              [=](Expr<LOGICAL> &&x, Expr<LOGICAL> &&y) {
                return Expr<LOGICAL>{LogicalOperation<KIND>{
                    operation.logicalOperator, std::move(x), std::move(y)}};
              }})}) {
    return *array;
  }
  if (auto folded{OperandsAreConstants(operation)}) {
    bool xt{folded->first.IsTrue()}, yt{folded->second.IsTrue()}, result{};
    switch (operation.logicalOperator) {
    case LogicalOperator::And:
      result = xt && yt;
      break;
    case LogicalOperator::Or:
      result = xt || yt;
      break;
    case LogicalOperator::Eqv:
      result = xt == yt;
      break;
    case LogicalOperator::Neqv:
      result = xt != yt;
      break;
    case LogicalOperator::Not:
      DIE("not a binary operator");
    }
    return Expr<LOGICAL>{Constant<LOGICAL>{result}};
  }
  return Expr<LOGICAL>{std::move(operation)};
}

#ifdef _MSC_VER // disable bogus warning about missing definitions
#pragma warning(disable : 4661)
#endif
FOR_EACH_LOGICAL_KIND(template class ExpressionBase, )
template class ExpressionBase<SomeLogical>;
} // namespace Fortran::evaluate
