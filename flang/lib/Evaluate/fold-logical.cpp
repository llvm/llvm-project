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
    exts.push_back(
        Scalar<LargestInt>::ConvertUnsigned(v, 8 * largestIntKind).value);
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
// TODO: unsigned

static Expr<SomeReal> RealToIntBound(
    int xRKind, int moldIKind, bool round, bool negate) {
  using RealType = Scalar<Type<TypeCategory::Real>>;
  using IntType = Scalar<Type<TypeCategory::Integer>>;
  RealType result{RealType::Zero(xRKind)}; // 0.
  common::RoundingMode roundingMode{round
          ? common::RoundingMode::TiesAwayFromZero
          : common::RoundingMode::ToZero};
  // Add decreasing powers of two to the result to find the largest magnitude
  // value that can be converted to the integer type without overflow.
  RealType at{
      RealType::FromInteger(IntType{negate ? -1 : 1, moldIKind}, xRKind).value};
  bool decrement{true};
  while (!at.ToInteger(roundingMode, 8 * moldIKind)
          .flags.test(RealFlag::Overflow)) {
    auto tmp{at.SCALE(IntType{1, moldIKind})};
    if (tmp.flags.test(RealFlag::Overflow)) {
      decrement = false;
      break;
    }
    at = tmp.value;
  }
  while (true) {
    if (decrement) {
      at = at.SCALE(IntType{-1, moldIKind}).value;
    } else {
      decrement = true;
    }
    auto tmp{at.Add(result)};
    if (tmp.flags.test(RealFlag::Inexact)) {
      break;
    } else if (!tmp.value.ToInteger(roundingMode, 8 * moldIKind)
                   .flags.test(RealFlag::Overflow)) {
      result = tmp.value;
    }
  }
  return AsCategoryExpr(Constant<Type<TypeCategory::Real>>{std::move(result)});
}

static std::optional<Expr<SomeReal>> RealToIntLimit(
    FoldingContext &context, Expr<SomeReal> &&hi, Expr<SomeReal> &lo) {
  using RealT = Type<TypeCategory::Real>;
  // The result kind is carried at runtime; iterate promoting to wider kinds
  // until the difference (hi - lo) can be represented exactly.
  while (true) {
    int kind{hi.GetType().value().kind()};
    bool promote{kind < 16};
    std::optional<Expr<SomeReal>> constResult;
    if (auto hiV{GetScalarConstantValue<RealT>(hi)}) {
      auto loV{GetScalarConstantValue<RealT>(lo)};
      CHECK(loV.has_value());
      auto diff{hiV->Subtract(*loV, Rounding{common::RoundingMode::ToZero})};
      promote = promote &&
          (diff.flags.test(RealFlag::Overflow) ||
              diff.flags.test(RealFlag::Inexact));
      constResult = AsCategoryExpr(Constant<RealT>{std::move(diff.value)});
    }
    if (promote) {
      int nextKind{kind < 4 ? 4 : kind == 4 ? 8 : 16};
      hi = Expr<SomeReal>{
          Fold(context, ConvertToType<RealT>(nextKind, std::move(hi)))};
      lo = Expr<SomeReal>{
          Fold(context, ConvertToType<RealT>(nextKind, std::move(lo)))};
      if (constResult) {
        // Recompute the difference at the promoted kind.
        continue;
      }
    }
    if (constResult) {
      return constResult;
    } else {
      return AsCategoryExpr(std::move(hi) - Expr<SomeReal>{lo});
    }
  }
}

// RealToRealBounds() returns a pair (HUGE(x),REAL(HUGE(mold),KIND(x)))
// when REAL(HUGE(x),KIND(mold)) overflows, and std::nullopt otherwise.
static std::optional<std::pair<Expr<SomeReal>, Expr<SomeReal>>>
RealToRealBounds(int xRKind, int moldRKind) {
  using RealType = Scalar<Type<TypeCategory::Real>>;
  if (!RealType::Convert(RealType::HUGE(xRKind), moldRKind)
          .flags.test(RealFlag::Overflow)) {
    return std::nullopt;
  } else {
    return std::make_pair(
        AsCategoryExpr(Constant<Type<TypeCategory::Real>>{
            RealType::Convert(RealType::HUGE(moldRKind), xRKind).value}),
        AsCategoryExpr(
            Constant<Type<TypeCategory::Real>>{RealType::HUGE(xRKind)}));
  }
}

static std::optional<Expr<SomeInteger>> IntToRealBound(
    int xIKind, int moldRKind, bool negate) {
  using IntType = Scalar<Type<TypeCategory::Integer>>;
  using RealType = Scalar<Type<TypeCategory::Real>>;
  IntType result{0, xIKind}; // 0
  while (true) {
    std::optional<IntType> next;
    for (int bit{0}; bit < 8 * xIKind; ++bit) {
      IntType power{IntType{0, xIKind}.IBSET(bit)};
      if (power.IsNegative()) {
        if (!negate) {
          break;
        }
      } else if (negate) {
        power = power.Negate().value;
      }
      auto tmp{power.AddSigned(result)};
      if (tmp.overflow ||
          RealType::FromInteger(tmp.value, moldRKind)
              .flags.test(RealFlag::Overflow)) {
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
  if (result.CompareSigned(IntType::HUGE(xIKind)) == Ordering::Equal) {
    return std::nullopt;
  } else {
    return AsCategoryExpr(
        Constant<Type<TypeCategory::Integer>>{std::move(result)});
  }
}

static std::optional<Expr<SomeInteger>> IntToIntBound(
    int xIKind, int moldIKind) {
  if (xIKind <= moldIKind) {
    return std::nullopt;
  }
  using IntegerType = Scalar<Type<TypeCategory::Integer>>;
  return AsCategoryExpr(Constant<Type<TypeCategory::Integer>>{
      IntegerType::ConvertSigned(IntegerType::HUGE(moldIKind), 8 * xIKind)
          .value});
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
    // The kind is carried at runtime by the procedure's result type, so only
    // the category needs to be matched among the lengthless intrinsic types.
    if (T::category == typeAndShape_->type().category()) {
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
      std::vector<Scalar<SubscriptInteger>>{
          Scalar<SubscriptInteger>{0, subscriptIntegerKind}},
      std::move(shape)};
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

static Expr<Type<TypeCategory::Logical>> RewriteOutOfRange(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Logical>> &&funcRef) {
  using ResultType = Type<TypeCategory::Logical>;
  const int resultKind{funcRef.GetType().value().kind()};
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
            auto one{DEREF(UnwrapExpr<Expr<SomeInteger>>(ConvertToType(xType,
                AsGenericExpr(Constant<SubscriptInteger>{
                    SubscriptInteger::Scalar{1, subscriptIntegerKind}}))))};
            auto lhs{std::move(*iXExpr) +
                (Expr<SomeInteger>{*hi} + Expr<SomeInteger>{one})};
            auto two{DEREF(UnwrapExpr<Expr<SomeInteger>>(ConvertToType(xType,
                AsGenericExpr(Constant<SubscriptInteger>{
                    SubscriptInteger::Scalar{2, subscriptIntegerKind}}))))};
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
              if (const Symbol *whole{UnwrapWholeSymbolDataRef(*round)};
                  whole && semantics::IsOptional(whole->GetUltimate())) {
                if (auto source{args[2]->sourceLocation()}) {
                  context.Warn(common::UsageWarning::OptionalMustBePresent,
                      *source,
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
                Expr<SomeInteger>{
                    Expr<SubscriptInteger>{Constant<SubscriptInteger>{
                        SubscriptInteger::Scalar{1, subscriptIntegerKind}}}}};
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
                  AsGenericExpr(Constant<SubscriptInteger>{
                      SubscriptInteger::Scalar{0, subscriptIntegerKind}}))}) {
            result = Expr<LogicalResult>{LogicalOperation{
                LogicalOperator::And, std::move(*nez), std::move(scalarFalse)}};
          }
        } else {
          result = std::move(scalarFalse);
        }
      }
      if (result) {
        auto restorer{context.messages().DiscardMessages()};
        return Fold(context,
            AsExpr(ConvertToType<ResultType>(resultKind, std::move(*result))));
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

Expr<Type<TypeCategory::Logical>> FoldIntrinsicFunction(FoldingContext &context,
    FunctionRef<Type<TypeCategory::Logical>> &&funcRef) {
  using T = Type<TypeCategory::Logical>;
  const int kind{funcRef.GetType().value().kind()};
  ActualArguments &args{funcRef.arguments()};
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  if (name == "all") {
    return FoldAllAnyParity(
        context, std::move(funcRef), &Scalar<T>::AND, Scalar<T>{true});
  } else if (name == "allocated") {
    if (IsNullAllocatable(args[0]->UnwrapExpr())) {
      return Expr<T>{false};
    }
  } else if (name == "any") {
    return FoldAllAnyParity(
        context, std::move(funcRef), &Scalar<T>::OR, Scalar<T>{false});
  } else if (name == "associated") {
    if (IsNullPointer(args[0]->UnwrapExpr()) ||
        (args[1] && IsNullPointer(args[1]->UnwrapExpr()))) {
      return Expr<T>{false};
    }
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
        // Copy rather than move: when only one operand is constant the fold
        // below is skipped and args[i] must retain its original BOZ value.
        constArgs[i] = AsGenericExpr(Constant<LargestInt>{*x});
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
    using SameInt = Type<TypeCategory::Integer>;
    if (const auto *ix{UnwrapExpr<Expr<SomeInteger>>(args[0])}) {
      return common::visit(
          [&](const auto &x) {
            using IT = ResultType<decltype(x)>;
            return FoldElementalIntrinsic<T, IT, SameInt>(context,
                std::move(funcRef),
                ScalarFunc<T, IT, SameInt>(
                    [&](const Scalar<IT> &x, const Scalar<SameInt> &pos) {
                      auto posVal{pos.ToInt64()};
                      if (posVal < 0 || posVal >= x.bits()) {
                        context.messages().Say(
                            "POS=%jd out of range for BTEST"_err_en_US,
                            static_cast<std::intmax_t>(posVal));
                      }
                      return Scalar<T>{x.BTEST(posVal)};
                    }));
          },
          ix->u);
    } else if (const auto *ux{UnwrapExpr<Expr<SomeUnsigned>>(args[0])}) {
      return common::visit(
          [&](const auto &x) {
            using UT = ResultType<decltype(x)>;
            return FoldElementalIntrinsic<T, UT, SameInt>(context,
                std::move(funcRef),
                ScalarFunc<T, UT, SameInt>(
                    [&](const Scalar<UT> &x, const Scalar<SameInt> &pos) {
                      auto posVal{pos.ToInt64()};
                      if (posVal < 0 || posVal >= x.bits()) {
                        context.messages().Say(
                            "POS=%jd out of range for BTEST"_err_en_US,
                            static_cast<std::intmax_t>(posVal));
                      }
                      return Scalar<T>{x.BTEST(posVal)};
                    }));
          },
          ux->u);
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
      using DefaultReal = Type<TypeCategory::Real>;
      return FoldElementalIntrinsic<T, DefaultReal>(context, std::move(funcRef),
          ScalarFunc<T, DefaultReal>([](const Scalar<DefaultReal> &x) {
            return Scalar<T>{x.IsNotANumber()};
          }));
    }
  } else if (name == "__builtin_ieee_is_negative") {
    auto restorer{context.messages().DiscardMessages()};
    using DefaultReal = Type<TypeCategory::Real>;
    if (args[0] && args[0]->UnwrapExpr() &&
        IsActuallyConstant(*args[0]->UnwrapExpr())) {
      return FoldElementalIntrinsic<T, DefaultReal>(context, std::move(funcRef),
          ScalarFunc<T, DefaultReal>([](const Scalar<DefaultReal> &x) {
            return Scalar<T>{x.IsNegative()};
          }));
    }
  } else if (name == "__builtin_ieee_is_normal") {
    auto restorer{context.messages().DiscardMessages()};
    using DefaultReal = Type<TypeCategory::Real>;
    if (args[0] && args[0]->UnwrapExpr() &&
        IsActuallyConstant(*args[0]->UnwrapExpr())) {
      return FoldElementalIntrinsic<T, DefaultReal>(context, std::move(funcRef),
          ScalarFunc<T, DefaultReal>([](const Scalar<DefaultReal> &x) {
            return Scalar<T>{x.IsNormal()};
          }));
    }
  } else if (name == "is_contiguous") {
    if (args.at(0)) {
      std::optional<bool> knownContiguous;
      if (auto *expr{args[0]->UnwrapExpr()}) {
        knownContiguous = IsContiguous(*expr, context);
      } else if (auto *assumedType{args[0]->GetAssumedTypeDummy()}) {
        knownContiguous = IsContiguous(*assumedType, context);
      }
      if (knownContiguous) {
        if (*knownContiguous) {
          if (auto source{args[0]->sourceLocation()}) {
            context.Warn(common::UsageWarning::ConstantIsContiguous, *source,
                "is_contiguous() is always true for named constants and subobjects of named constants"_warn_en_US);
          }
        }
        return Expr<T>{*knownContiguous};
      }
    }
  } else if (name == "is_iostat_end") {
    if (args[0] && args[0]->UnwrapExpr() &&
        IsActuallyConstant(*args[0]->UnwrapExpr())) {
      using Int64 = Type<TypeCategory::Integer>;
      return FoldElementalIntrinsic<T, Int64>(context, std::move(funcRef),
          ScalarFunc<T, Int64>([](const Scalar<Int64> &x) {
            return Scalar<T>{x.ToInt64() == FORTRAN_RUNTIME_IOSTAT_END};
          }));
    }
  } else if (name == "is_iostat_eor") {
    if (args[0] && args[0]->UnwrapExpr() &&
        IsActuallyConstant(*args[0]->UnwrapExpr())) {
      using Int64 = Type<TypeCategory::Integer>;
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
          ConvertToType<T>(kind,
              PackageRelation(name == "lge" ? RelationalOperator::GE
                      : name == "lgt"       ? RelationalOperator::GT
                      : name == "lle"       ? RelationalOperator::LE
                                            : RelationalOperator::LT,
                  ConvertToType<Ascii>(asciiKind, std::move(*cx0)),
                  ConvertToType<Ascii>(asciiKind, std::move(*cx1)))));
    }
  } else if (name == "logical") {
    if (auto *expr{UnwrapExpr<Expr<SomeLogical>>(args[0])}) {
      return Fold(context, ConvertToType<T>(kind, std::move(*expr)));
    }
  } else if (name == "matmul") {
    return FoldMatmul(context, std::move(funcRef));
  } else if (name == "out_of_range") {
    return RewriteOutOfRange(context, std::move(funcRef));
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
    if (context.targetCharacteristics().ieeeFeatures().test(
            IeeeFeature::Flags)) {
      if (args[0]) {
        if (const auto *cst{UnwrapExpr<Constant<SomeDerived>>(args[0])}) {
          if (auto constr{cst->GetScalarValue()}) {
            if (StructureConstructorValues & values{constr->values()};
                values.size() == 1) {
              const Expr<SomeType> &value{values.begin()->second.value()};
              if (auto flag{ToInt64(value)}) {
                if (flag != _FORTRAN_RUNTIME_IEEE_DENORM) {
                  // Check for suppport for standard exceptions.
                  return Expr<T>{
                      context.targetCharacteristics().ieeeFeatures().test(
                          IeeeFeature::Flags)};
                } else if (args[1]) {
                  // Check for nonstandard ieee_denorm exception support for
                  // a given kind.
                  return Expr<T>{context.targetCharacteristics()
                          .hasSubnormalExceptionSupport(
                              args[1]->GetType().value().kind())};
                } else {
                  // Check for nonstandard ieee_denorm exception support for
                  // all kinds.
                  return Expr<T>{context.targetCharacteristics()
                          .hasSubnormalExceptionSupport()};
                }
              }
            }
          }
        }
      }
    }
  } else if (name == "__builtin_ieee_support_halting") {
    if (!context.targetCharacteristics()
            .haltingSupportIsUnknownAtCompileTime()) {
      return Expr<T>{context.targetCharacteristics().ieeeFeatures().test(
          IeeeFeature::Halting)};
    }
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
        return Expr<T>{mode < common::RoundingMode::TiesAwayFromZero};
      }
    }
  } else if (name == "__builtin_ieee_support_sqrt") {
    return Expr<T>{
        context.targetCharacteristics().ieeeFeatures().test(IeeeFeature::Sqrt)};
  } else if (name == "__builtin_ieee_support_standard") {
    // ieee_support_standard depends in part on ieee_support_halting.
    if (!context.targetCharacteristics()
            .haltingSupportIsUnknownAtCompileTime()) {
      return Expr<T>{context.targetCharacteristics().ieeeFeatures().test(
          IeeeFeature::Standard)};
    }
  } else if (name == "__builtin_ieee_support_subnormal") {
    return Expr<T>{context.targetCharacteristics().ieeeFeatures().test(
        IeeeFeature::Subnormal)};
  } else if (name == "__builtin_ieee_support_underflow_control") {
    // Setting kind=0 checks subnormal flushing control across all type kinds.
    if (args[0]) {
      return Expr<T>{
          context.targetCharacteristics().hasSubnormalFlushingControl(
              args[0]->GetType().value().kind())};
    } else {
      return Expr<T>{
          context.targetCharacteristics().hasSubnormalFlushingControl(
              /*any=*/false)};
    }
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
    } else if constexpr (T::category == TypeCategory::Unsigned) {
      result = Satisfies(
          relation.opr, folded->first.CompareUnsigned(folded->second));
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

Expr<Type<TypeCategory::Logical>> FoldOperation(
    FoldingContext &context, Not &&x) {
  if (auto array{ApplyElementwise(context, x)}) {
    return *array;
  }
  using Ty = Type<TypeCategory::Logical>;
  auto &operand{x.left()};
  if (auto value{GetScalarConstantValue<Ty>(operand)}) {
    return Expr<Ty>{Constant<Ty>{!value->IsTrue()}};
  }
  return Expr<Ty>{x};
}

Expr<Type<TypeCategory::Logical>> FoldOperation(
    FoldingContext &context, LogicalOperation &&operation) {
  using LOGICAL = Type<TypeCategory::Logical>;
  if (auto array{ApplyElementwise(context, operation,
          std::function<Expr<LOGICAL>(Expr<LOGICAL> &&, Expr<LOGICAL> &&)>{
              [=](Expr<LOGICAL> &&x, Expr<LOGICAL> &&y) {
                return Expr<LOGICAL>{LogicalOperation{
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
