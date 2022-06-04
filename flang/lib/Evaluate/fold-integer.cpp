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
    if (int rank{array->Rank()}; rank > 0) {
      std::optional<int> dim;
      if (funcRef.Rank() == 0) {
        // Optional DIM= argument is present: result is scalar.
        if (auto dim64{GetInt64Arg(args[1])}) {
          if (*dim64 < 1 || *dim64 > rank) {
            context.messages().Say("DIM=%jd dimension is out of range for "
                                   "rank-%d array"_err_en_US,
                *dim64, rank);
            return MakeInvalidIntrinsic<T>(std::move(funcRef));
          } else {
            dim = *dim64 - 1; // 1-based to 0-based
          }
        } else {
          // DIM= is present but not constant
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
    if (int rank{array->Rank()}; rank > 0) {
      std::optional<int> dim;
      if (funcRef.Rank() == 0) {
        // Optional DIM= argument is present: result is scalar.
        if (auto dim64{GetInt64Arg(args[1])}) {
          if (*dim64 < 1 || *dim64 > rank) {
            context.messages().Say("DIM=%jd dimension is out of range for "
                                   "rank-%d array"_err_en_US,
                *dim64, rank);
            return MakeInvalidIntrinsic<T>(std::move(funcRef));
          } else {
            dim = *dim64 - 1; // 1-based to 0-based
          }
        } else {
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
            if (semantics::IsAssumedSizeArray(symbol) && *dim == rank - 1) {
              context.messages().Say("DIM=%jd dimension is out of range for "
                                     "rank-%d assumed-size array"_err_en_US,
                  rank, rank);
              return MakeInvalidIntrinsic<T>(std::move(funcRef));
            } else if (auto ub{GetUBOUND(context, *named, *dim)}) {
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
template <typename T>
static Expr<T> FoldCount(FoldingContext &context, FunctionRef<T> &&ref) {
  static_assert(T::category == TypeCategory::Integer);
  ActualArguments &arg{ref.arguments()};
  if (const Constant<LogicalResult> *mask{arg.empty()
              ? nullptr
              : Folder<LogicalResult>{context}.Folding(arg[0])}) {
    std::optional<int> dim;
    if (CheckReductionDIM(dim, context, arg, 1, mask->Rank())) {
      auto accumulator{[&](Scalar<T> &element, const ConstantSubscripts &at) {
        if (mask->At(at).IsTrue()) {
          element = element.AddSigned(Scalar<T>{1}).value;
        }
      }};
      return Expr<T>{DoReduction<T>(*mask, dim, Scalar<T>{}, accumulator)};
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
    return FoldCount<T>(context, std::move(funcRef));
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
    return FoldElementalIntrinsic<T, T, T>(
        context, std::move(funcRef), &Scalar<T>::DIM);
  } else if (name == "dshiftl" || name == "dshiftr") {
    const auto fptr{
        name == "dshiftl" ? &Scalar<T>::DSHIFTL : &Scalar<T>::DSHIFTR};
    // Third argument can be of any kind. However, it must be smaller or equal
    // than BIT_SIZE. It can be converted to Int4 to simplify.
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
            [&funcRef, &context](const auto &str) -> Expr<T> {
              using Char = typename std::decay_t<decltype(str)>::Result;
              return FoldElementalIntrinsic<T, Char>(context,
                  std::move(funcRef),
                  ScalarFunc<T, Char>([](const Scalar<Char> &c) {
                    return Scalar<T>{CharacterUtils<Char::kind>::ICHAR(c)};
                  }));
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
    return FoldElementalIntrinsic<T, T, Int4>(context, std::move(funcRef),
        ScalarFunc<T, T, Int4>([&](const Scalar<T> &i,
                                   const Scalar<Int4> &pos) -> Scalar<T> {
          auto posVal{static_cast<int>(pos.ToInt64())};
          if (posVal < 0) {
            context.messages().Say(
                "bit position for %s (%d) is negative"_err_en_US, name, posVal);
          } else if (posVal >= i.bits) {
            context.messages().Say(
                "bit position for %s (%d) is not less than %d"_err_en_US, name,
                posVal, i.bits);
          }
          return std::invoke(fptr, i, posVal);
        }));
  } else if (name == "ibits") {
    return FoldElementalIntrinsic<T, T, Int4, Int4>(context, std::move(funcRef),
        ScalarFunc<T, T, Int4, Int4>([&](const Scalar<T> &i,
                                         const Scalar<Int4> &pos,
                                         const Scalar<Int4> &len) -> Scalar<T> {
          auto posVal{static_cast<int>(pos.ToInt64())};
          auto lenVal{static_cast<int>(len.ToInt64())};
          if (posVal < 0) {
            context.messages().Say(
                "bit position for IBITS(POS=%d,LEN=%d) is negative"_err_en_US,
                posVal, lenVal);
          } else if (lenVal < 0) {
            context.messages().Say(
                "bit length for IBITS(POS=%d,LEN=%d) is negative"_err_en_US,
                posVal, lenVal);
          } else if (posVal + lenVal > i.bits) {
            context.messages().Say(
                "IBITS(POS=%d,LEN=%d) must have POS+LEN no greater than %d"_err_en_US,
                posVal + lenVal, i.bits);
          }
          return i.IBITS(posVal, lenVal);
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
                      [&name](const Scalar<TC> &str, const Scalar<TC> &other,
                          const Scalar<LogicalResult> &back) -> Scalar<T> {
                        return name == "index"
                            ? CharacterUtils<TC::kind>::INDEX(
                                  str, other, back.IsTrue())
                            : name == "scan" ? CharacterUtils<TC::kind>::SCAN(
                                                   str, other, back.IsTrue())
                                             : CharacterUtils<TC::kind>::VERIFY(
                                                   str, other, back.IsTrue());
                      }});
            } else {
              return FoldElementalIntrinsic<T, TC, TC>(context,
                  std::move(funcRef),
                  ScalarFunc<T, TC, TC>{
                      [&name](const Scalar<TC> &str,
                          const Scalar<TC> &other) -> Scalar<T> {
                        return name == "index"
                            ? CharacterUtils<TC::kind>::INDEX(str, other)
                            : name == "scan"
                            ? CharacterUtils<TC::kind>::SCAN(str, other)
                            : CharacterUtils<TC::kind>::VERIFY(str, other);
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
  } else if (name == "ishft") {
    return FoldElementalIntrinsic<T, T, Int4>(context, std::move(funcRef),
        ScalarFunc<T, T, Int4>([&](const Scalar<T> &i,
                                   const Scalar<Int4> &pos) -> Scalar<T> {
          auto posVal{static_cast<int>(pos.ToInt64())};
          if (posVal < -i.bits) {
            context.messages().Say(
                "SHIFT=%d count for ishft is less than %d"_err_en_US, posVal,
                -i.bits);
          } else if (posVal > i.bits) {
            context.messages().Say(
                "SHIFT=%d count for ishft is greater than %d"_err_en_US, posVal,
                i.bits);
          }
          return i.ISHFT(posVal);
        }));
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
                ScalarFunc<T, TI>([&fptr](const Scalar<TI> &i) -> Scalar<T> {
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
                ScalarFunc<T, TC>{[](const Scalar<TC> &str) -> Scalar<T> {
                  return CharacterUtils<TC::kind>::LEN_TRIM(str);
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
        ScalarFuncWithContext<T, T, T>(
            [](FoldingContext &context, const Scalar<T> &x,
                const Scalar<T> &y) -> Scalar<T> {
              auto result{x.MODULO(y)};
              if (result.overflow) {
                context.messages().Say(
                    "modulo() folding overflowed"_warn_en_US);
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
    if (auto p{GetInt64Arg(args[0])}) {
      return Expr<T>{SelectedIntKind(*p)};
    }
  } else if (name == "selected_real_kind" ||
      name == "__builtin_ieee_selected_real_kind") {
    if (auto p{GetInt64ArgOr(args[0], 0)}) {
      if (auto r{GetInt64ArgOr(args[1], 0)}) {
        if (auto radix{GetInt64ArgOr(args[2], 2)}) {
          return Expr<T>{SelectedRealKind(*p, *r, *radix)};
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
    return FoldElementalIntrinsic<T, T, Int4>(context, std::move(funcRef),
        ScalarFunc<T, T, Int4>([&](const Scalar<T> &i,
                                   const Scalar<Int4> &pos) -> Scalar<T> {
          auto posVal{static_cast<int>(pos.ToInt64())};
          if (posVal < 0) {
            context.messages().Say(
                "SHIFT=%d count for %s is negative"_err_en_US, posVal, name);
          } else if (posVal > i.bits) {
            context.messages().Say(
                "SHIFT=%d count for %s is greater than %d"_err_en_US, posVal,
                name, i.bits);
          }
          return std::invoke(fptr, i, posVal);
        }));
  } else if (name == "sign") {
    return FoldElementalIntrinsic<T, T, T>(context, std::move(funcRef),
        ScalarFunc<T, T, T>(
            [&context](const Scalar<T> &j, const Scalar<T> &k) -> Scalar<T> {
              typename Scalar<T>::ValueWithOverflow result{j.SIGN(k)};
              if (result.overflow) {
                context.messages().Say(
                    "sign(integer(kind=%d)) folding overflowed"_warn_en_US,
                    KIND);
              }
              return result.value;
            }));
  } else if (name == "size") {
    if (auto shape{GetContextFreeShape(context, args[0])}) {
      if (auto &dimArg{args[1]}) { // DIM= is present, get one extent
        if (auto dim{GetInt64Arg(args[1])}) {
          int rank{GetRank(*shape)};
          if (*dim >= 1 && *dim <= rank) {
            const Symbol *symbol{UnwrapWholeSymbolDataRef(args[0])};
            if (symbol && IsAssumedSizeArray(*symbol) && *dim == rank) {
              context.messages().Say(
                  "size(array,dim=%jd) of last dimension is not available for rank-%d assumed-size array dummy argument"_err_en_US,
                  *dim, rank);
              return MakeInvalidIntrinsic<T>(std::move(funcRef));
            } else if (auto &extent{shape->at(*dim - 1)}) {
              return Fold(context, ConvertToType<T>(std::move(*extent)));
            }
          } else {
            context.messages().Say(
                "size(array,dim=%jd) dimension is out of range for rank-%d array"_warn_en_US,
                *dim, rank);
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
  // TODO: dot_product, ishftc, matmul, sign, transfer
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
    // in a current derived type instantiation, for KIND type parameters.
    if (const auto *pdt{context.pdtInstance()}) {
      bool isLen{false};
      if (const semantics::Scope * scope{context.pdtInstance()->scope()}) {
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
  if (const auto *intExpr{UnwrapExpr<Expr<SomeInteger>>(expr)}) {
    return ToInt64(*intExpr);
  } else {
    return std::nullopt;
  }
}

#ifdef _MSC_VER // disable bogus warning about missing definitions
#pragma warning(disable : 4661)
#endif
FOR_EACH_INTEGER_KIND(template class ExpressionBase, )
template class ExpressionBase<SomeInteger>;
} // namespace Fortran::evaluate
