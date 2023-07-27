//===-- lib/Evaluate/fold-integer.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "fold-implementation.h"
#include "fold-reduction.h"
#include "flang/Evaluate/check-expression.h"

namespace Fortran::evaluate {

// Given a collection of ConstantSubscripts values, package them as a Constant.
// Return scalar value if asScalar == true and shape-dim array otherwise.
template <typename T>
Expr<T> PackageConstantBounds(
    const ConstantSubscripts &&bounds, bool asScalar = false) {
  if (asScalar) {
    return Expr<T>{Constant<T>{bounds.at(0)}};
  } else {
    // As rank-dim array
    const int rank{GetRank(bounds)};
    std::vector<Scalar<T>> packed(rank);
    std::transform(bounds.begin(), bounds.end(), packed.begin(),
        [](ConstantSubscript x) { return Scalar<T>(x); });
    return Expr<T>{Constant<T>{std::move(packed), ConstantSubscripts{rank}}};
  }
}

// If a DIM= argument to LBOUND(), UBOUND(), or SIZE() exists and has a valid
// constant value, return in "dimVal" that value, less 1 (to make it suitable
// for use as a C++ vector<> index).  Also check for erroneous constant values
// and returns false on error.
static bool CheckDimArg(const std::optional<ActualArgument> &dimArg,
    const Expr<SomeType> &array, parser::ContextualMessages &messages,
    bool isLBound, std::optional<int> &dimVal) {
  dimVal.reset();
  if (int rank{array.Rank()}; rank > 0 || IsAssumedRank(array)) {
    auto named{ExtractNamedEntity(array)};
    if (auto dim64{ToInt64(dimArg)}) {
      if (*dim64 < 1) {
        messages.Say("DIM=%jd dimension must be positive"_err_en_US, *dim64);
        return false;
      } else if (!IsAssumedRank(array) && *dim64 > rank) {
        messages.Say(
            "DIM=%jd dimension is out of range for rank-%d array"_err_en_US,
            *dim64, rank);
        return false;
      } else if (!isLBound && named &&
          semantics::IsAssumedSizeArray(named->GetLastSymbol()) &&
          *dim64 == rank) {
        messages.Say(
            "DIM=%jd dimension is out of range for rank-%d assumed-size array"_err_en_US,
            *dim64, rank);
        return false;
      } else if (IsAssumedRank(array)) {
        if (*dim64 > common::maxRank) {
          messages.Say(
              "DIM=%jd dimension is too large for any array (maximum rank %d)"_err_en_US,
              *dim64, common::maxRank);
          return false;
        }
      } else {
        dimVal = static_cast<int>(*dim64 - 1); // 1-based to 0-based
      }
    }
  }
  return true;
}

// Class to retrieve the constant bound of an expression which is an
// array that devolves to a type of Constant<T>
class GetConstantArrayBoundHelper {
public:
  template <typename T>
  static Expr<T> GetLbound(
      const Expr<SomeType> &array, std::optional<int> dim) {
    return PackageConstantBounds<T>(
        GetConstantArrayBoundHelper(dim, /*getLbound=*/true).Get(array),
        dim.has_value());
  }

  template <typename T>
  static Expr<T> GetUbound(
      const Expr<SomeType> &array, std::optional<int> dim) {
    return PackageConstantBounds<T>(
        GetConstantArrayBoundHelper(dim, /*getLbound=*/false).Get(array),
        dim.has_value());
  }

private:
  GetConstantArrayBoundHelper(
      std::optional<ConstantSubscript> dim, bool getLbound)
      : dim_{dim}, getLbound_{getLbound} {}

  template <typename T> ConstantSubscripts Get(const T &) {
    // The method is needed for template expansion, but we should never get
    // here in practice.
    CHECK(false);
    return {0};
  }

  template <typename T> ConstantSubscripts Get(const Constant<T> &x) {
    if (getLbound_) {
      // Return the lower bound
      if (dim_) {
        return {x.lbounds().at(*dim_)};
      } else {
        return x.lbounds();
      }
    } else {
      // Return the upper bound
      if (arrayFromParenthesesExpr) {
        // Underlying array comes from (x) expression - return shapes
        if (dim_) {
          return {x.shape().at(*dim_)};
        } else {
          return x.shape();
        }
      } else {
        return x.ComputeUbounds(dim_);
      }
    }
  }

  template <typename T> ConstantSubscripts Get(const Parentheses<T> &x) {
    // Cause of temp variable inside parentheses - return [1, ... 1] for lower
    // bounds and shape for upper bounds
    if (getLbound_) {
      return ConstantSubscripts(x.Rank(), ConstantSubscript{1});
    } else {
      // Indicate that underlying array comes from parentheses expression.
      // Continue to unwrap expression until we hit a constant
      arrayFromParenthesesExpr = true;
      return Get(x.left());
    }
  }

  template <typename T> ConstantSubscripts Get(const Expr<T> &x) {
    // recurse through Expr<T>'a until we hit a constant
    return common::visit([&](const auto &inner) { return Get(inner); },
        //      [&](const auto &) { return 0; },
        x.u);
  }

  const std::optional<ConstantSubscript> dim_;
  const bool getLbound_;
  bool arrayFromParenthesesExpr{false};
};

template <int KIND>
Expr<Type<TypeCategory::Integer, KIND>> LBOUND(FoldingContext &context,
    FunctionRef<Type<TypeCategory::Integer, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Integer, KIND>;
  ActualArguments &args{funcRef.arguments()};
  if (const auto *array{UnwrapExpr<Expr<SomeType>>(args[0])}) {
    if (int rank{array->Rank()}; rank > 0 || IsAssumedRank(*array)) {
      std::optional<int> dim;
      if (funcRef.Rank() == 0) {
        // Optional DIM= argument is present: result is scalar.
        if (!CheckDimArg(args[1], *array, context.messages(), true, dim)) {
          return MakeInvalidIntrinsic<T>(std::move(funcRef));
        } else if (!dim) {
          // DIM= is present but not constant, or error
          return Expr<T>{std::move(funcRef)};
        }
      }
      bool lowerBoundsAreOne{true};
      if (auto named{ExtractNamedEntity(*array)}) {
        const Symbol &symbol{named->GetLastSymbol()};
        if (symbol.Rank() == rank) {
          lowerBoundsAreOne = false;
          if (dim) {
            if (auto lb{GetLBOUND(context, *named, *dim)}) {
              return Fold(context, ConvertToType<T>(std::move(*lb)));
            }
          } else if (auto extents{
                         AsExtentArrayExpr(GetLBOUNDs(context, *named))}) {
            return Fold(context,
                ConvertToType<T>(Expr<ExtentType>{std::move(*extents)}));
          }
        } else {
          lowerBoundsAreOne = symbol.Rank() == 0; // LBOUND(array%component)
        }
      }
      if (IsActuallyConstant(*array)) {
        return GetConstantArrayBoundHelper::GetLbound<T>(*array, dim);
      }
      if (lowerBoundsAreOne) {
        ConstantSubscripts ones(rank, ConstantSubscript{1});
        return PackageConstantBounds<T>(std::move(ones), dim.has_value());
      }
    }
  }
  return Expr<T>{std::move(funcRef)};
}

template <int KIND>
Expr<Type<TypeCategory::Integer, KIND>> UBOUND(FoldingContext &context,
    FunctionRef<Type<TypeCategory::Integer, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Integer, KIND>;
  ActualArguments &args{funcRef.arguments()};
  if (auto *array{UnwrapExpr<Expr<SomeType>>(args[0])}) {
    if (int rank{array->Rank()}; rank > 0 || IsAssumedRank(*array)) {
      std::optional<int> dim;
      if (funcRef.Rank() == 0) {
        // Optional DIM= argument is present: result is scalar.
        if (!CheckDimArg(args[1], *array, context.messages(), false, dim)) {
          return MakeInvalidIntrinsic<T>(std::move(funcRef));
        } else if (!dim) {
          // DIM= is present but not constant
          return Expr<T>{std::move(funcRef)};
        }
      }
      bool takeBoundsFromShape{true};
      if (auto named{ExtractNamedEntity(*array)}) {
        const Symbol &symbol{named->GetLastSymbol()};
        if (symbol.Rank() == rank) {
          takeBoundsFromShape = false;
          if (dim) {
            if (auto ub{GetUBOUND(context, *named, *dim)}) {
              return Fold(context, ConvertToType<T>(std::move(*ub)));
            }
          } else {
            Shape ubounds{GetUBOUNDs(context, *named)};
            if (semantics::IsAssumedSizeArray(symbol)) {
              CHECK(!ubounds.back());
              ubounds.back() = ExtentExpr{-1};
            }
            if (auto extents{AsExtentArrayExpr(ubounds)}) {
              return Fold(context,
                  ConvertToType<T>(Expr<ExtentType>{std::move(*extents)}));
            }
          }
        } else {
          takeBoundsFromShape = symbol.Rank() == 0; // UBOUND(array%component)
        }
      }
      if (IsActuallyConstant(*array)) {
        return GetConstantArrayBoundHelper::GetUbound<T>(*array, dim);
      }
      if (takeBoundsFromShape) {
        if (auto shape{GetContextFreeShape(context, *array)}) {
          if (dim) {
            if (auto &dimSize{shape->at(*dim)}) {
              return Fold(context,
                  ConvertToType<T>(Expr<ExtentType>{std::move(*dimSize)}));
            }
          } else if (auto shapeExpr{AsExtentArrayExpr(*shape)}) {
            return Fold(context, ConvertToType<T>(std::move(*shapeExpr)));
          }
        }
      }
    }
  }
  return Expr<T>{std::move(funcRef)};
}

// COUNT()
template <typename T, int maskKind>
static Expr<T> FoldCount(FoldingContext &context, FunctionRef<T> &&ref) {
  using LogicalResult = Type<TypeCategory::Logical, maskKind>;
  static_assert(T::category == TypeCategory::Integer);
  ActualArguments &arg{ref.arguments()};
  if (const Constant<LogicalResult> *mask{arg.empty()
              ? nullptr
              : Folder<LogicalResult>{context}.Folding(arg[0])}) {
    std::optional<int> dim;
    if (CheckReductionDIM(dim, context, arg, 1, mask->Rank())) {
      bool overflow{false};
      auto accumulator{
          [&mask, &overflow](Scalar<T> &element, const ConstantSubscripts &at) {
            if (mask->At(at).IsTrue()) {
              auto incremented{element.AddSigned(Scalar<T>{1})};
              overflow |= incremented.overflow;
              element = incremented.value;
            }
          }};
      Constant<T> result{DoReduction<T>(*mask, dim, Scalar<T>{}, accumulator)};
      if (overflow) {
        context.messages().Say(
            "Result of intrinsic function COUNT overflows its result type"_warn_en_US);
      }
      return Expr<T>{std::move(result)};
    }
  }
  return Expr<T>{std::move(ref)};
}

// FINDLOC(), MAXLOC(), & MINLOC()
enum class WhichLocation { Findloc, Maxloc, Minloc };
template <WhichLocation WHICH> class LocationHelper {
public:
  LocationHelper(
      DynamicType &&type, ActualArguments &arg, FoldingContext &context)
      : type_{type}, arg_{arg}, context_{context} {}
  using Result = std::optional<Constant<SubscriptInteger>>;
  using Types = std::conditional_t<WHICH == WhichLocation::Findloc,
      AllIntrinsicTypes, RelationalTypes>;

  template <typename T> Result Test() const {
    if (T::category != type_.category() || T::kind != type_.kind()) {
      return std::nullopt;
    }
    CHECK(arg_.size() == (WHICH == WhichLocation::Findloc ? 6 : 5));
    Folder<T> folder{context_};
    Constant<T> *array{folder.Folding(arg_[0])};
    if (!array) {
      return std::nullopt;
    }
    std::optional<Constant<T>> value;
    if constexpr (WHICH == WhichLocation::Findloc) {
      if (const Constant<T> *p{folder.Folding(arg_[1])}) {
        value.emplace(*p);
      } else {
        return std::nullopt;
      }
    }
    std::optional<int> dim;
    Constant<LogicalResult> *mask{
        GetReductionMASK(arg_[maskArg], array->shape(), context_)};
    if ((!mask && arg_[maskArg]) ||
        !CheckReductionDIM(dim, context_, arg_, dimArg, array->Rank())) {
      return std::nullopt;
    }
    bool back{false};
    if (arg_[backArg]) {
      const auto *backConst{
          Folder<LogicalResult>{context_}.Folding(arg_[backArg])};
      if (backConst) {
        back = backConst->GetScalarValue().value().IsTrue();
      } else {
        return std::nullopt;
      }
    }
    const RelationalOperator relation{WHICH == WhichLocation::Findloc
            ? RelationalOperator::EQ
            : WHICH == WhichLocation::Maxloc
            ? (back ? RelationalOperator::GE : RelationalOperator::GT)
            : back ? RelationalOperator::LE
                   : RelationalOperator::LT};
    // Use lower bounds of 1 exclusively.
    array->SetLowerBoundsToOne();
    ConstantSubscripts at{array->lbounds()}, maskAt, resultIndices, resultShape;
    if (mask) {
      if (auto scalarMask{mask->GetScalarValue()}) {
        // Convert into array in case of scalar MASK= (for
        // MAXLOC/MINLOC/FINDLOC mask should be be conformable)
        ConstantSubscript n{GetSize(array->shape())};
        std::vector<Scalar<LogicalResult>> mask_elements(
            n, Scalar<LogicalResult>{scalarMask.value()});
        *mask = Constant<LogicalResult>{
            std::move(mask_elements), ConstantSubscripts{n}};
      }
      mask->SetLowerBoundsToOne();
      maskAt = mask->lbounds();
    }
    if (dim) { // DIM=
      if (*dim < 1 || *dim > array->Rank()) {
        context_.messages().Say("DIM=%d is out of range"_err_en_US, *dim);
        return std::nullopt;
      }
      int zbDim{*dim - 1};
      resultShape = array->shape();
      resultShape.erase(
          resultShape.begin() + zbDim); // scalar if array is vector
      ConstantSubscript dimLength{array->shape()[zbDim]};
      ConstantSubscript n{GetSize(resultShape)};
      for (ConstantSubscript j{0}; j < n; ++j) {
        ConstantSubscript hit{0};
        if constexpr (WHICH == WhichLocation::Maxloc ||
            WHICH == WhichLocation::Minloc) {
          value.reset();
        }
        for (ConstantSubscript k{0}; k < dimLength;
             ++k, ++at[zbDim], mask && ++maskAt[zbDim]) {
          if ((!mask || mask->At(maskAt).IsTrue()) &&
              IsHit(array->At(at), value, relation)) {
            hit = at[zbDim];
            if constexpr (WHICH == WhichLocation::Findloc) {
              if (!back) {
                break;
              }
            }
          }
        }
        resultIndices.emplace_back(hit);
        at[zbDim] = std::max<ConstantSubscript>(dimLength, 1);
        array->IncrementSubscripts(at);
        at[zbDim] = 1;
        if (mask) {
          maskAt[zbDim] = mask->lbounds()[zbDim] +
              std::max<ConstantSubscript>(dimLength, 1) - 1;
          mask->IncrementSubscripts(maskAt);
          maskAt[zbDim] = mask->lbounds()[zbDim];
        }
      }
    } else { // no DIM=
      resultShape = ConstantSubscripts{array->Rank()}; // always a vector
      ConstantSubscript n{GetSize(array->shape())};
      resultIndices = ConstantSubscripts(array->Rank(), 0);
      for (ConstantSubscript j{0}; j < n; ++j, array->IncrementSubscripts(at),
           mask && mask->IncrementSubscripts(maskAt)) {
        if ((!mask || mask->At(maskAt).IsTrue()) &&
            IsHit(array->At(at), value, relation)) {
          resultIndices = at;
          if constexpr (WHICH == WhichLocation::Findloc) {
            if (!back) {
              break;
            }
          }
        }
      }
    }
    std::vector<Scalar<SubscriptInteger>> resultElements;
    for (ConstantSubscript j : resultIndices) {
      resultElements.emplace_back(j);
    }
    return Constant<SubscriptInteger>{
        std::move(resultElements), std::move(resultShape)};
  }

private:
  template <typename T>
  bool IsHit(typename Constant<T>::Element element,
      std::optional<Constant<T>> &value,
      [[maybe_unused]] RelationalOperator relation) const {
    std::optional<Expr<LogicalResult>> cmp;
    bool result{true};
    if (value) {
      if constexpr (T::category == TypeCategory::Logical) {
        // array(at) .EQV. value?
        static_assert(WHICH == WhichLocation::Findloc);
        cmp.emplace(ConvertToType<LogicalResult>(
            Expr<T>{LogicalOperation<T::kind>{LogicalOperator::Eqv,
                Expr<T>{Constant<T>{element}}, Expr<T>{Constant<T>{*value}}}}));
      } else { // compare array(at) to value
        cmp.emplace(PackageRelation(relation, Expr<T>{Constant<T>{element}},
            Expr<T>{Constant<T>{*value}}));
      }
      Expr<LogicalResult> folded{Fold(context_, std::move(*cmp))};
      result = GetScalarConstantValue<LogicalResult>(folded).value().IsTrue();
    } else {
      // first unmasked element for MAXLOC/MINLOC - always take it
    }
    if constexpr (WHICH == WhichLocation::Maxloc ||
        WHICH == WhichLocation::Minloc) {
      if (result) {
        value.emplace(std::move(element));
      }
    }
    return result;
  }

  static constexpr int dimArg{WHICH == WhichLocation::Findloc ? 2 : 1};
  static constexpr int maskArg{dimArg + 1};
  static constexpr int backArg{maskArg + 2};

  DynamicType type_;
  ActualArguments &arg_;
  FoldingContext &context_;
};

template <WhichLocation which>
static std::optional<Constant<SubscriptInteger>> FoldLocationCall(
    ActualArguments &arg, FoldingContext &context) {
  if (arg[0]) {
    if (auto type{arg[0]->GetType()}) {
      if constexpr (which == WhichLocation::Findloc) {
        // Both ARRAY and VALUE are susceptible to conversion to a common
        // comparison type.
        if (arg[1]) {
          if (auto valType{arg[1]->GetType()}) {
            if (auto compareType{ComparisonType(*type, *valType)}) {
              type = compareType;
            }
          }
        }
      }
      return common::SearchTypes(
          LocationHelper<which>{std::move(*type), arg, context});
    }
  }
  return std::nullopt;
}

template <WhichLocation which, typename T>
static Expr<T> FoldLocation(FoldingContext &context, FunctionRef<T> &&ref) {
  static_assert(T::category == TypeCategory::Integer);
  if (std::optional<Constant<SubscriptInteger>> found{
          FoldLocationCall<which>(ref.arguments(), context)}) {
    return Expr<T>{Fold(
        context, ConvertToType<T>(Expr<SubscriptInteger>{std::move(*found)}))};
  } else {
    return Expr<T>{std::move(ref)};
  }
}

// for IALL, IANY, & IPARITY
template <typename T>
static Expr<T> FoldBitReduction(FoldingContext &context, FunctionRef<T> &&ref,
    Scalar<T> (Scalar<T>::*operation)(const Scalar<T> &) const,
    Scalar<T> identity) {
  static_assert(T::category == TypeCategory::Integer);
  std::optional<int> dim;
  if (std::optional<Constant<T>> array{
          ProcessReductionArgs<T>(context, ref.arguments(), dim, identity,
              /*ARRAY=*/0, /*DIM=*/1, /*MASK=*/2)}) {
    auto accumulator{[&](Scalar<T> &element, const ConstantSubscripts &at) {
      element = (element.*operation)(array->At(at));
    }};
    return Expr<T>{DoReduction<T>(*array, dim, identity, accumulator)};
  }
  return Expr<T>{std::move(ref)};
}

template <int KIND>
Expr<Type<TypeCategory::Integer, KIND>> FoldIntrinsicFunction(
    FoldingContext &context,
    FunctionRef<Type<TypeCategory::Integer, KIND>> &&funcRef) {
  using T = Type<TypeCategory::Integer, KIND>;
  using Int4 = Type<TypeCategory::Integer, 4>;
  ActualArguments &args{funcRef.arguments()};
  auto *intrinsic{std::get_if<SpecificIntrinsic>(&funcRef.proc().u)};
  CHECK(intrinsic);
  std::string name{intrinsic->name};
  auto FromInt64{[&name, &context](std::int64_t n) {
    Scalar<T> result{n};
    if (result.ToInt64() != n) {
      context.messages().Say(
          "Result of intrinsic function '%s' (%jd) overflows its result type"_warn_en_US,
          name, std::intmax_t{n});
    }
    return result;
  }};
  if (name == "abs") { // incl. babs, iiabs, jiaabs, & kiabs
    return FoldElementalIntrinsic<T, T>(context, std::move(funcRef),
        ScalarFunc<T, T>([&context](const Scalar<T> &i) -> Scalar<T> {
          typename Scalar<T>::ValueWithOverflow j{i.ABS()};
          if (j.overflow) {
            context.messages().Say(
                "abs(integer(kind=%d)) folding overflowed"_warn_en_US, KIND);
          }
          return j.value;
        }));
  } else if (name == "bit_size") {
    return Expr<T>{Scalar<T>::bits};
  } else if (name == "ceiling" || name == "floor" || name == "nint") {
    if (const auto *cx{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      // NINT rounds ties away from zero, not to even
      common::RoundingMode mode{name == "ceiling" ? common::RoundingMode::Up
              : name == "floor"                   ? common::RoundingMode::Down
                                : common::RoundingMode::TiesAwayFromZero};
      return common::visit(
          [&](const auto &kx) {
            using TR = ResultType<decltype(kx)>;
            return FoldElementalIntrinsic<T, TR>(context, std::move(funcRef),
                ScalarFunc<T, TR>([&](const Scalar<TR> &x) {
                  auto y{x.template ToInteger<Scalar<T>>(mode)};
                  if (y.flags.test(RealFlag::Overflow)) {
                    context.messages().Say(
                        "%s intrinsic folding overflow"_warn_en_US, name);
                  }
                  return y.value;
                }));
          },
          cx->u);
    }
  } else if (name == "count") {
    int maskKind = args[0]->GetType()->kind();
    switch (maskKind) {
      SWITCH_COVERS_ALL_CASES
    case 1:
      return FoldCount<T, 1>(context, std::move(funcRef));
    case 2:
      return FoldCount<T, 2>(context, std::move(funcRef));
    case 4:
      return FoldCount<T, 4>(context, std::move(funcRef));
    case 8:
      return FoldCount<T, 8>(context, std::move(funcRef));
    }
  } else if (name == "digits") {
    if (const auto *cx{UnwrapExpr<Expr<SomeInteger>>(args[0])}) {
      return Expr<T>{common::visit(
          [](const auto &kx) {
            return Scalar<ResultType<decltype(kx)>>::DIGITS;
          },
          cx->u)};
    } else if (const auto *cx{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      return Expr<T>{common::visit(
          [](const auto &kx) {
            return Scalar<ResultType<decltype(kx)>>::DIGITS;
          },
          cx->u)};
    } else if (const auto *cx{UnwrapExpr<Expr<SomeComplex>>(args[0])}) {
      return Expr<T>{common::visit(
          [](const auto &kx) {
            return Scalar<typename ResultType<decltype(kx)>::Part>::DIGITS;
          },
          cx->u)};
    }
  } else if (name == "dim") {
    return FoldElementalIntrinsic<T, T, T>(context, std::move(funcRef),
        ScalarFunc<T, T, T>([&context](const Scalar<T> &x,
                                const Scalar<T> &y) -> Scalar<T> {
          auto result{x.DIM(y)};
          if (result.overflow) {
            context.messages().Say("DIM intrinsic folding overflow"_warn_en_US);
          }
          return result.value;
        }));
  } else if (name == "dot_product") {
    return FoldDotProduct<T>(context, std::move(funcRef));
  } else if (name == "dshiftl" || name == "dshiftr") {
    const auto fptr{
        name == "dshiftl" ? &Scalar<T>::DSHIFTL : &Scalar<T>::DSHIFTR};
    // Third argument can be of any kind. However, it must be smaller or equal
    // than BIT_SIZE. It can be converted to Int4 to simplify.
    if (const auto *argCon{Folder<T>(context).Folding(args[0])};
        argCon && argCon->empty()) {
    } else if (const auto *shiftCon{Folder<Int4>(context).Folding(args[2])}) {
      for (const auto &scalar : shiftCon->values()) {
        std::int64_t shiftVal{scalar.ToInt64()};
        if (shiftVal < 0) {
          context.messages().Say("SHIFT=%jd count for %s is negative"_err_en_US,
              std::intmax_t{shiftVal}, name);
          break;
        } else if (shiftVal > T::Scalar::bits) {
          context.messages().Say(
              "SHIFT=%jd count for %s is greater than %d"_err_en_US,
              std::intmax_t{shiftVal}, name, T::Scalar::bits);
          break;
        }
      }
    }
    return FoldElementalIntrinsic<T, T, T, Int4>(context, std::move(funcRef),
        ScalarFunc<T, T, T, Int4>(
            [&fptr](const Scalar<T> &i, const Scalar<T> &j,
                const Scalar<Int4> &shift) -> Scalar<T> {
              return std::invoke(fptr, i, j, static_cast<int>(shift.ToInt64()));
            }));
  } else if (name == "exponent") {
    if (auto *sx{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      return common::visit(
          [&funcRef, &context](const auto &x) -> Expr<T> {
            using TR = typename std::decay_t<decltype(x)>::Result;
            return FoldElementalIntrinsic<T, TR>(context, std::move(funcRef),
                &Scalar<TR>::template EXPONENT<Scalar<T>>);
          },
          sx->u);
    } else {
      DIE("exponent argument must be real");
    }
  } else if (name == "findloc") {
    return FoldLocation<WhichLocation::Findloc, T>(context, std::move(funcRef));
  } else if (name == "huge") {
    return Expr<T>{Scalar<T>::HUGE()};
  } else if (name == "iachar" || name == "ichar") {
    auto *someChar{UnwrapExpr<Expr<SomeCharacter>>(args[0])};
    CHECK(someChar);
    if (auto len{ToInt64(someChar->LEN())}) {
      if (len.value() != 1) {
        // Do not die, this was not checked before
        context.messages().Say(
            "Character in intrinsic function %s must have length one"_warn_en_US,
            name);
      } else {
        return common::visit(
            [&funcRef, &context, &FromInt64](const auto &str) -> Expr<T> {
              using Char = typename std::decay_t<decltype(str)>::Result;
              return FoldElementalIntrinsic<T, Char>(context,
                  std::move(funcRef),
                  ScalarFunc<T, Char>(
#ifndef _MSC_VER
                      [&FromInt64](const Scalar<Char> &c) {
                        return FromInt64(CharacterUtils<Char::kind>::ICHAR(c));
                      }));
#else // _MSC_VER
      // MSVC 14 get confused by the original code above and
      // ends up emitting an error about passing a std::string
      // to the std::u16string instantiation of
      // CharacterUtils<2>::ICHAR(). Can't find a work-around,
      // so remove the FromInt64 error checking lambda that
      // seems to have caused the proble.
                      [](const Scalar<Char> &c) {
                        return CharacterUtils<Char::kind>::ICHAR(c);
                      }));
#endif // _MSC_VER
            },
            someChar->u);
      }
    }
  } else if (name == "iand" || name == "ior" || name == "ieor") {
    auto fptr{&Scalar<T>::IAND};
    if (name == "iand") { // done in fptr declaration
    } else if (name == "ior") {
      fptr = &Scalar<T>::IOR;
    } else if (name == "ieor") {
      fptr = &Scalar<T>::IEOR;
    } else {
      common::die("missing case to fold intrinsic function %s", name.c_str());
    }
    return FoldElementalIntrinsic<T, T, T>(
        context, std::move(funcRef), ScalarFunc<T, T, T>(fptr));
  } else if (name == "iall") {
    return FoldBitReduction(
        context, std::move(funcRef), &Scalar<T>::IAND, Scalar<T>{}.NOT());
  } else if (name == "iany") {
    return FoldBitReduction(
        context, std::move(funcRef), &Scalar<T>::IOR, Scalar<T>{});
  } else if (name == "ibclr" || name == "ibset") {
    // Second argument can be of any kind. However, it must be smaller
    // than BIT_SIZE. It can be converted to Int4 to simplify.
    auto fptr{&Scalar<T>::IBCLR};
    if (name == "ibclr") { // done in fptr definition
    } else if (name == "ibset") {
      fptr = &Scalar<T>::IBSET;
    } else {
      common::die("missing case to fold intrinsic function %s", name.c_str());
    }
    if (const auto *argCon{Folder<T>(context).Folding(args[0])};
        argCon && argCon->empty()) {
    } else if (const auto *posCon{Folder<Int4>(context).Folding(args[1])}) {
      for (const auto &scalar : posCon->values()) {
        std::int64_t posVal{scalar.ToInt64()};
        if (posVal < 0) {
          context.messages().Say(
              "bit position for %s (%jd) is negative"_err_en_US, name,
              std::intmax_t{posVal});
          break;
        } else if (posVal >= T::Scalar::bits) {
          context.messages().Say(
              "bit position for %s (%jd) is not less than %d"_err_en_US, name,
              std::intmax_t{posVal}, T::Scalar::bits);
          break;
        }
      }
    }
    return FoldElementalIntrinsic<T, T, Int4>(context, std::move(funcRef),
        ScalarFunc<T, T, Int4>(
            [&](const Scalar<T> &i, const Scalar<Int4> &pos) -> Scalar<T> {
              return std::invoke(fptr, i, static_cast<int>(pos.ToInt64()));
            }));
  } else if (name == "ibits") {
    const auto *posCon{Folder<Int4>(context).Folding(args[1])};
    const auto *lenCon{Folder<Int4>(context).Folding(args[2])};
    if (const auto *argCon{Folder<T>(context).Folding(args[0])};
        argCon && argCon->empty()) {
    } else {
      std::size_t posCt{posCon ? posCon->size() : 0};
      std::size_t lenCt{lenCon ? lenCon->size() : 0};
      std::size_t n{std::max(posCt, lenCt)};
      for (std::size_t j{0}; j < n; ++j) {
        int posVal{j < posCt || posCt == 1
                ? static_cast<int>(posCon->values()[j % posCt].ToInt64())
                : 0};
        int lenVal{j < lenCt || lenCt == 1
                ? static_cast<int>(lenCon->values()[j % lenCt].ToInt64())
                : 0};
        if (posVal < 0) {
          context.messages().Say(
              "bit position for IBITS(POS=%jd) is negative"_err_en_US,
              std::intmax_t{posVal});
          break;
        } else if (lenVal < 0) {
          context.messages().Say(
              "bit length for IBITS(LEN=%jd) is negative"_err_en_US,
              std::intmax_t{lenVal});
          break;
        } else if (posVal + lenVal > T::Scalar::bits) {
          context.messages().Say(
              "IBITS() must have POS+LEN (>=%jd) no greater than %d"_err_en_US,
              std::intmax_t{posVal + lenVal}, T::Scalar::bits);
          break;
        }
      }
    }
    return FoldElementalIntrinsic<T, T, Int4, Int4>(context, std::move(funcRef),
        ScalarFunc<T, T, Int4, Int4>(
            [&](const Scalar<T> &i, const Scalar<Int4> &pos,
                const Scalar<Int4> &len) -> Scalar<T> {
              return i.IBITS(static_cast<int>(pos.ToInt64()),
                  static_cast<int>(len.ToInt64()));
            }));
  } else if (name == "index" || name == "scan" || name == "verify") {
    if (auto *charExpr{UnwrapExpr<Expr<SomeCharacter>>(args[0])}) {
      return common::visit(
          [&](const auto &kch) -> Expr<T> {
            using TC = typename std::decay_t<decltype(kch)>::Result;
            if (UnwrapExpr<Expr<SomeLogical>>(args[2])) { // BACK=
              return FoldElementalIntrinsic<T, TC, TC, LogicalResult>(context,
                  std::move(funcRef),
                  ScalarFunc<T, TC, TC, LogicalResult>{
                      [&name, &FromInt64](const Scalar<TC> &str,
                          const Scalar<TC> &other,
                          const Scalar<LogicalResult> &back) {
                        return FromInt64(name == "index"
                                ? CharacterUtils<TC::kind>::INDEX(
                                      str, other, back.IsTrue())
                                : name == "scan"
                                ? CharacterUtils<TC::kind>::SCAN(
                                      str, other, back.IsTrue())
                                : CharacterUtils<TC::kind>::VERIFY(
                                      str, other, back.IsTrue()));
                      }});
            } else {
              return FoldElementalIntrinsic<T, TC, TC>(context,
                  std::move(funcRef),
                  ScalarFunc<T, TC, TC>{
                      [&name, &FromInt64](
                          const Scalar<TC> &str, const Scalar<TC> &other) {
                        return FromInt64(name == "index"
                                ? CharacterUtils<TC::kind>::INDEX(str, other)
                                : name == "scan"
                                ? CharacterUtils<TC::kind>::SCAN(str, other)
                                : CharacterUtils<TC::kind>::VERIFY(str, other));
                      }});
            }
          },
          charExpr->u);
    } else {
      DIE("first argument must be CHARACTER");
    }
  } else if (name == "int") {
    if (auto *expr{UnwrapExpr<Expr<SomeType>>(args[0])}) {
      return common::visit(
          [&](auto &&x) -> Expr<T> {
            using From = std::decay_t<decltype(x)>;
            if constexpr (std::is_same_v<From, BOZLiteralConstant> ||
                IsNumericCategoryExpr<From>()) {
              return Fold(context, ConvertToType<T>(std::move(x)));
            }
            DIE("int() argument type not valid");
          },
          std::move(expr->u));
    }
  } else if (name == "int_ptr_kind") {
    return Expr<T>{8};
  } else if (name == "kind") {
    if constexpr (common::HasMember<T, IntegerTypes>) {
      return Expr<T>{args[0].value().GetType()->kind()};
    } else {
      DIE("kind() result not integral");
    }
  } else if (name == "iparity") {
    return FoldBitReduction(
        context, std::move(funcRef), &Scalar<T>::IEOR, Scalar<T>{});
  } else if (name == "ishft" || name == "ishftc") {
    const auto *argCon{Folder<T>(context).Folding(args[0])};
    const auto *shiftCon{Folder<Int4>(context).Folding(args[1])};
    const auto *shiftVals{shiftCon ? &shiftCon->values() : nullptr};
    const auto *sizeCon{
        args.size() == 3 ? Folder<Int4>(context).Folding(args[2]) : nullptr};
    const auto *sizeVals{sizeCon ? &sizeCon->values() : nullptr};
    if ((argCon && argCon->empty()) || !shiftVals || shiftVals->empty() ||
        (sizeVals && sizeVals->empty())) {
      // size= and shift= values don't need to be checked
    } else {
      for (const auto &scalar : *shiftVals) {
        std::int64_t shiftVal{scalar.ToInt64()};
        if (shiftVal < -T::Scalar::bits) {
          context.messages().Say(
              "SHIFT=%jd count for %s is less than %d"_err_en_US,
              std::intmax_t{shiftVal}, name, -T::Scalar::bits);
          break;
        } else if (shiftVal > T::Scalar::bits) {
          context.messages().Say(
              "SHIFT=%jd count for %s is greater than %d"_err_en_US,
              std::intmax_t{shiftVal}, name, T::Scalar::bits);
          break;
        }
      }
      if (sizeVals) {
        for (const auto &scalar : *sizeVals) {
          std::int64_t sizeVal{scalar.ToInt64()};
          if (sizeVal <= 0) {
            context.messages().Say(
                "SIZE=%jd count for ishftc is not positive"_err_en_US,
                std::intmax_t{sizeVal}, name);
            break;
          } else if (sizeVal > T::Scalar::bits) {
            context.messages().Say(
                "SIZE=%jd count for ishftc is greater than %d"_err_en_US,
                std::intmax_t{sizeVal}, T::Scalar::bits);
            break;
          }
        }
        if (shiftVals->size() == 1 || sizeVals->size() == 1 ||
            shiftVals->size() == sizeVals->size()) {
          auto iters{std::max(shiftVals->size(), sizeVals->size())};
          for (std::size_t j{0}; j < iters; ++j) {
            auto shiftVal{static_cast<int>(
                (*shiftVals)[j % shiftVals->size()].ToInt64())};
            auto sizeVal{
                static_cast<int>((*sizeVals)[j % sizeVals->size()].ToInt64())};
            if (sizeVal > 0 && std::abs(shiftVal) > sizeVal) {
              context.messages().Say(
                  "SHIFT=%jd count for ishftc is greater in magnitude than SIZE=%jd"_err_en_US,
                  std::intmax_t{shiftVal}, std::intmax_t{sizeVal});
              break;
            }
          }
        }
      }
    }
    if (name == "ishft") {
      return FoldElementalIntrinsic<T, T, Int4>(context, std::move(funcRef),
          ScalarFunc<T, T, Int4>(
              [&](const Scalar<T> &i, const Scalar<Int4> &shift) -> Scalar<T> {
                return i.ISHFT(static_cast<int>(shift.ToInt64()));
              }));
    } else if (!args.at(2)) { // ISHFTC(no SIZE=)
      return FoldElementalIntrinsic<T, T, Int4>(context, std::move(funcRef),
          ScalarFunc<T, T, Int4>(
              [&](const Scalar<T> &i, const Scalar<Int4> &shift) -> Scalar<T> {
                return i.ISHFTC(static_cast<int>(shift.ToInt64()));
              }));
    } else { // ISHFTC(with SIZE=)
      return FoldElementalIntrinsic<T, T, Int4, Int4>(context,
          std::move(funcRef),
          ScalarFunc<T, T, Int4, Int4>(
              [&](const Scalar<T> &i, const Scalar<Int4> &shift,
                  const Scalar<Int4> &size) -> Scalar<T> {
                auto shiftVal{static_cast<int>(shift.ToInt64())};
                auto sizeVal{static_cast<int>(size.ToInt64())};
                return i.ISHFTC(shiftVal, sizeVal);
              }));
    }
  } else if (name == "izext" || name == "jzext") {
    if (args.size() == 1) {
      if (auto *expr{UnwrapExpr<Expr<SomeInteger>>(args[0])}) {
        // Rewrite to IAND(INT(n,k),255_k) for k=KIND(T)
        intrinsic->name = "iand";
        auto converted{ConvertToType<T>(std::move(*expr))};
        *expr = Fold(context, Expr<SomeInteger>{std::move(converted)});
        args.emplace_back(AsGenericExpr(Expr<T>{Scalar<T>{255}}));
        return FoldIntrinsicFunction(context, std::move(funcRef));
      }
    }
  } else if (name == "lbound") {
    return LBOUND(context, std::move(funcRef));
  } else if (name == "leadz" || name == "trailz" || name == "poppar" ||
      name == "popcnt") {
    if (auto *sn{UnwrapExpr<Expr<SomeInteger>>(args[0])}) {
      return common::visit(
          [&funcRef, &context, &name](const auto &n) -> Expr<T> {
            using TI = typename std::decay_t<decltype(n)>::Result;
            if (name == "poppar") {
              return FoldElementalIntrinsic<T, TI>(context, std::move(funcRef),
                  ScalarFunc<T, TI>([](const Scalar<TI> &i) -> Scalar<T> {
                    return Scalar<T>{i.POPPAR() ? 1 : 0};
                  }));
            }
            auto fptr{&Scalar<TI>::LEADZ};
            if (name == "leadz") { // done in fptr definition
            } else if (name == "trailz") {
              fptr = &Scalar<TI>::TRAILZ;
            } else if (name == "popcnt") {
              fptr = &Scalar<TI>::POPCNT;
            } else {
              common::die(
                  "missing case to fold intrinsic function %s", name.c_str());
            }
            return FoldElementalIntrinsic<T, TI>(context, std::move(funcRef),
                // `i` should be declared as `const Scalar<TI>&`.
                // We declare it as `auto` to workaround an msvc bug:
                // https://developercommunity.visualstudio.com/t/Regression:-nested-closure-assumes-wrong/10130223
                ScalarFunc<T, TI>([&fptr](const auto &i) -> Scalar<T> {
                  return Scalar<T>{std::invoke(fptr, i)};
                }));
          },
          sn->u);
    } else {
      DIE("leadz argument must be integer");
    }
  } else if (name == "len") {
    if (auto *charExpr{UnwrapExpr<Expr<SomeCharacter>>(args[0])}) {
      return common::visit(
          [&](auto &kx) {
            if (auto len{kx.LEN()}) {
              if (IsScopeInvariantExpr(*len)) {
                return Fold(context, ConvertToType<T>(*std::move(len)));
              } else {
                return Expr<T>{std::move(funcRef)};
              }
            } else {
              return Expr<T>{std::move(funcRef)};
            }
          },
          charExpr->u);
    } else {
      DIE("len() argument must be of character type");
    }
  } else if (name == "len_trim") {
    if (auto *charExpr{UnwrapExpr<Expr<SomeCharacter>>(args[0])}) {
      return common::visit(
          [&](const auto &kch) -> Expr<T> {
            using TC = typename std::decay_t<decltype(kch)>::Result;
            return FoldElementalIntrinsic<T, TC>(context, std::move(funcRef),
                ScalarFunc<T, TC>{[&FromInt64](const Scalar<TC> &str) {
                  return FromInt64(CharacterUtils<TC::kind>::LEN_TRIM(str));
                }});
          },
          charExpr->u);
    } else {
      DIE("len_trim() argument must be of character type");
    }
  } else if (name == "maskl" || name == "maskr") {
    // Argument can be of any kind but value has to be smaller than BIT_SIZE.
    // It can be safely converted to Int4 to simplify.
    const auto fptr{name == "maskl" ? &Scalar<T>::MASKL : &Scalar<T>::MASKR};
    return FoldElementalIntrinsic<T, Int4>(context, std::move(funcRef),
        ScalarFunc<T, Int4>([&fptr](const Scalar<Int4> &places) -> Scalar<T> {
          return fptr(static_cast<int>(places.ToInt64()));
        }));
  } else if (name == "max") {
    return FoldMINorMAX(context, std::move(funcRef), Ordering::Greater);
  } else if (name == "max0" || name == "max1") {
    return RewriteSpecificMINorMAX(context, std::move(funcRef));
  } else if (name == "maxexponent") {
    if (auto *sx{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      return common::visit(
          [](const auto &x) {
            using TR = typename std::decay_t<decltype(x)>::Result;
            return Expr<T>{Scalar<TR>::MAXEXPONENT};
          },
          sx->u);
    }
  } else if (name == "maxloc") {
    return FoldLocation<WhichLocation::Maxloc, T>(context, std::move(funcRef));
  } else if (name == "maxval") {
    return FoldMaxvalMinval<T>(context, std::move(funcRef),
        RelationalOperator::GT, T::Scalar::Least());
  } else if (name == "merge") {
    return FoldMerge<T>(context, std::move(funcRef));
  } else if (name == "merge_bits") {
    return FoldElementalIntrinsic<T, T, T, T>(
        context, std::move(funcRef), &Scalar<T>::MERGE_BITS);
  } else if (name == "min") {
    return FoldMINorMAX(context, std::move(funcRef), Ordering::Less);
  } else if (name == "min0" || name == "min1") {
    return RewriteSpecificMINorMAX(context, std::move(funcRef));
  } else if (name == "minexponent") {
    if (auto *sx{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      return common::visit(
          [](const auto &x) {
            using TR = typename std::decay_t<decltype(x)>::Result;
            return Expr<T>{Scalar<TR>::MINEXPONENT};
          },
          sx->u);
    }
  } else if (name == "minloc") {
    return FoldLocation<WhichLocation::Minloc, T>(context, std::move(funcRef));
  } else if (name == "minval") {
    return FoldMaxvalMinval<T>(
        context, std::move(funcRef), RelationalOperator::LT, T::Scalar::HUGE());
  } else if (name == "mod") {
    return FoldElementalIntrinsic<T, T, T>(context, std::move(funcRef),
        ScalarFuncWithContext<T, T, T>(
            [](FoldingContext &context, const Scalar<T> &x,
                const Scalar<T> &y) -> Scalar<T> {
              auto quotRem{x.DivideSigned(y)};
              if (quotRem.divisionByZero) {
                context.messages().Say("mod() by zero"_warn_en_US);
              } else if (quotRem.overflow) {
                context.messages().Say("mod() folding overflowed"_warn_en_US);
              }
              return quotRem.remainder;
            }));
  } else if (name == "modulo") {
    return FoldElementalIntrinsic<T, T, T>(context, std::move(funcRef),
        ScalarFuncWithContext<T, T, T>([](FoldingContext &context,
                                           const Scalar<T> &x,
                                           const Scalar<T> &y) -> Scalar<T> {
          auto result{x.MODULO(y)};
          if (result.overflow) {
            context.messages().Say("modulo() folding overflowed"_warn_en_US);
          }
          return result.value;
        }));
  } else if (name == "not") {
    return FoldElementalIntrinsic<T, T>(
        context, std::move(funcRef), &Scalar<T>::NOT);
  } else if (name == "precision") {
    if (const auto *cx{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      return Expr<T>{common::visit(
          [](const auto &kx) {
            return Scalar<ResultType<decltype(kx)>>::PRECISION;
          },
          cx->u)};
    } else if (const auto *cx{UnwrapExpr<Expr<SomeComplex>>(args[0])}) {
      return Expr<T>{common::visit(
          [](const auto &kx) {
            return Scalar<typename ResultType<decltype(kx)>::Part>::PRECISION;
          },
          cx->u)};
    }
  } else if (name == "product") {
    return FoldProduct<T>(context, std::move(funcRef), Scalar<T>{1});
  } else if (name == "radix") {
    return Expr<T>{2};
  } else if (name == "range") {
    if (const auto *cx{UnwrapExpr<Expr<SomeInteger>>(args[0])}) {
      return Expr<T>{common::visit(
          [](const auto &kx) {
            return Scalar<ResultType<decltype(kx)>>::RANGE;
          },
          cx->u)};
    } else if (const auto *cx{UnwrapExpr<Expr<SomeReal>>(args[0])}) {
      return Expr<T>{common::visit(
          [](const auto &kx) {
            return Scalar<ResultType<decltype(kx)>>::RANGE;
          },
          cx->u)};
    } else if (const auto *cx{UnwrapExpr<Expr<SomeComplex>>(args[0])}) {
      return Expr<T>{common::visit(
          [](const auto &kx) {
            return Scalar<typename ResultType<decltype(kx)>::Part>::RANGE;
          },
          cx->u)};
    }
  } else if (name == "rank") {
    if (const auto *array{UnwrapExpr<Expr<SomeType>>(args[0])}) {
      if (auto named{ExtractNamedEntity(*array)}) {
        const Symbol &symbol{named->GetLastSymbol()};
        if (IsAssumedRank(symbol)) {
          // DescriptorInquiry can only be placed in expression of kind
          // DescriptorInquiry::Result::kind.
          return ConvertToType<T>(Expr<
              Type<TypeCategory::Integer, DescriptorInquiry::Result::kind>>{
              DescriptorInquiry{*named, DescriptorInquiry::Field::Rank}});
        }
      }
      return Expr<T>{args[0].value().Rank()};
    }
    return Expr<T>{args[0].value().Rank()};
  } else if (name == "selected_char_kind") {
    if (const auto *chCon{UnwrapExpr<Constant<TypeOf<std::string>>>(args[0])}) {
      if (std::optional<std::string> value{chCon->GetScalarValue()}) {
        int defaultKind{
            context.defaults().GetDefaultKind(TypeCategory::Character)};
        return Expr<T>{SelectedCharKind(*value, defaultKind)};
      }
    }
  } else if (name == "selected_int_kind") {
    if (auto p{ToInt64(args[0])}) {
      return Expr<T>{context.targetCharacteristics().SelectedIntKind(*p)};
    }
  } else if (name == "selected_real_kind" ||
      name == "__builtin_ieee_selected_real_kind") {
    if (auto p{GetInt64ArgOr(args[0], 0)}) {
      if (auto r{GetInt64ArgOr(args[1], 0)}) {
        if (auto radix{GetInt64ArgOr(args[2], 2)}) {
          return Expr<T>{
              context.targetCharacteristics().SelectedRealKind(*p, *r, *radix)};
        }
      }
    }
  } else if (name == "shape") {
    if (auto shape{GetContextFreeShape(context, args[0])}) {
      if (auto shapeExpr{AsExtentArrayExpr(*shape)}) {
        return Fold(context, ConvertToType<T>(std::move(*shapeExpr)));
      }
    }
  } else if (name == "shifta" || name == "shiftr" || name == "shiftl") {
    // Second argument can be of any kind. However, it must be smaller or
    // equal than BIT_SIZE. It can be converted to Int4 to simplify.
    auto fptr{&Scalar<T>::SHIFTA};
    if (name == "shifta") { // done in fptr definition
    } else if (name == "shiftr") {
      fptr = &Scalar<T>::SHIFTR;
    } else if (name == "shiftl") {
      fptr = &Scalar<T>::SHIFTL;
    } else {
      common::die("missing case to fold intrinsic function %s", name.c_str());
    }
    if (const auto *argCon{Folder<T>(context).Folding(args[0])};
        argCon && argCon->empty()) {
    } else if (const auto *shiftCon{Folder<Int4>(context).Folding(args[1])}) {
      for (const auto &scalar : shiftCon->values()) {
        std::int64_t shiftVal{scalar.ToInt64()};
        if (shiftVal < 0) {
          context.messages().Say("SHIFT=%jd count for %s is negative"_err_en_US,
              std::intmax_t{shiftVal}, name, -T::Scalar::bits);
          break;
        } else if (shiftVal > T::Scalar::bits) {
          context.messages().Say(
              "SHIFT=%jd count for %s is greater than %d"_err_en_US,
              std::intmax_t{shiftVal}, name, T::Scalar::bits);
          break;
        }
      }
    }
    return FoldElementalIntrinsic<T, T, Int4>(context, std::move(funcRef),
        ScalarFunc<T, T, Int4>(
            [&](const Scalar<T> &i, const Scalar<Int4> &shift) -> Scalar<T> {
              return std::invoke(fptr, i, static_cast<int>(shift.ToInt64()));
            }));
  } else if (name == "sign") {
    return FoldElementalIntrinsic<T, T, T>(context, std::move(funcRef),
        ScalarFunc<T, T, T>([&context](const Scalar<T> &j,
                                const Scalar<T> &k) -> Scalar<T> {
          typename Scalar<T>::ValueWithOverflow result{j.SIGN(k)};
          if (result.overflow) {
            context.messages().Say(
                "sign(integer(kind=%d)) folding overflowed"_warn_en_US, KIND);
          }
          return result.value;
        }));
  } else if (name == "size") {
    if (auto shape{GetContextFreeShape(context, args[0])}) {
      if (args[1]) { // DIM= is present, get one extent
        std::optional<int> dim;
        if (const auto *array{args[0].value().UnwrapExpr()}; array &&
            !CheckDimArg(args[1], *array, context.messages(), false, dim)) {
          return MakeInvalidIntrinsic<T>(std::move(funcRef));
        } else if (dim) {
          if (auto &extent{shape->at(*dim)}) {
            return Fold(context, ConvertToType<T>(std::move(*extent)));
          }
        }
      } else if (auto extents{common::AllElementsPresent(std::move(*shape))}) {
        // DIM= is absent; compute PRODUCT(SHAPE())
        ExtentExpr product{1};
        for (auto &&extent : std::move(*extents)) {
          product = std::move(product) * std::move(extent);
        }
        return Expr<T>{ConvertToType<T>(Fold(context, std::move(product)))};
      }
    }
  } else if (name == "sizeof") { // in bytes; extension
    if (auto info{
            characteristics::TypeAndShape::Characterize(args[0], context)}) {
      if (auto bytes{info->MeasureSizeInBytes(context)}) {
        return Expr<T>{Fold(context, ConvertToType<T>(std::move(*bytes)))};
      }
    }
  } else if (name == "storage_size") { // in bits
    if (auto info{
            characteristics::TypeAndShape::Characterize(args[0], context)}) {
      if (auto bytes{info->MeasureElementSizeInBytes(context, true)}) {
        return Expr<T>{
            Fold(context, Expr<T>{8} * ConvertToType<T>(std::move(*bytes)))};
      }
    }
  } else if (name == "sum") {
    return FoldSum<T>(context, std::move(funcRef));
  } else if (name == "ubound") {
    return UBOUND(context, std::move(funcRef));
  }
  // TODO: dot_product, matmul, sign
  return Expr<T>{std::move(funcRef)};
}

// Substitutes a bare type parameter reference with its value if it has one now
// in an instantiation.  Bare LEN type parameters are substituted only when
// the known value is constant.
Expr<TypeParamInquiry::Result> FoldOperation(
    FoldingContext &context, TypeParamInquiry &&inquiry) {
  std::optional<NamedEntity> base{inquiry.base()};
  parser::CharBlock parameterName{inquiry.parameter().name()};
  if (base) {
    // Handling "designator%typeParam".  Get the value of the type parameter
    // from the instantiation of the base
    if (const semantics::DeclTypeSpec *
        declType{base->GetLastSymbol().GetType()}) {
      if (const semantics::ParamValue *
          paramValue{
              declType->derivedTypeSpec().FindParameter(parameterName)}) {
        const semantics::MaybeIntExpr &paramExpr{paramValue->GetExplicit()};
        if (paramExpr && IsConstantExpr(*paramExpr)) {
          Expr<SomeInteger> intExpr{*paramExpr};
          return Fold(context,
              ConvertToType<TypeParamInquiry::Result>(std::move(intExpr)));
        }
      }
    }
  } else {
    // A "bare" type parameter: replace with its value, if that's now known
    // in a current derived type instantiation.
    if (const auto *pdt{context.pdtInstance()}) {
      auto restorer{context.WithoutPDTInstance()}; // don't loop
      bool isLen{false};
      if (const semantics::Scope * scope{pdt->scope()}) {
        auto iter{scope->find(parameterName)};
        if (iter != scope->end()) {
          const Symbol &symbol{*iter->second};
          const auto *details{symbol.detailsIf<semantics::TypeParamDetails>()};
          if (details) {
            isLen = details->attr() == common::TypeParamAttr::Len;
            const semantics::MaybeIntExpr &initExpr{details->init()};
            if (initExpr && IsConstantExpr(*initExpr) &&
                (!isLen || ToInt64(*initExpr))) {
              Expr<SomeInteger> expr{*initExpr};
              return Fold(context,
                  ConvertToType<TypeParamInquiry::Result>(std::move(expr)));
            }
          }
        }
      }
      if (const auto *value{pdt->FindParameter(parameterName)}) {
        if (value->isExplicit()) {
          auto folded{Fold(context,
              AsExpr(ConvertToType<TypeParamInquiry::Result>(
                  Expr<SomeInteger>{value->GetExplicit().value()})))};
          if (!isLen || ToInt64(folded)) {
            return folded;
          }
        }
      }
    }
  }
  return AsExpr(std::move(inquiry));
}

std::optional<std::int64_t> ToInt64(const Expr<SomeInteger> &expr) {
  return common::visit(
      [](const auto &kindExpr) { return ToInt64(kindExpr); }, expr.u);
}

std::optional<std::int64_t> ToInt64(const Expr<SomeType> &expr) {
  return ToInt64(UnwrapExpr<Expr<SomeInteger>>(expr));
}

std::optional<std::int64_t> ToInt64(const ActualArgument &arg) {
  return ToInt64(arg.UnwrapExpr());
}

#ifdef _MSC_VER // disable bogus warning about missing definitions
#pragma warning(disable : 4661)
#endif
FOR_EACH_INTEGER_KIND(template class ExpressionBase, )
template class ExpressionBase<SomeInteger>;
} // namespace Fortran::evaluate
