//===-- lib/Evaluate/fold-reduction.h -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_FOLD_REDUCTION_H_
#define FORTRAN_EVALUATE_FOLD_REDUCTION_H_

#include "fold-implementation.h"

namespace Fortran::evaluate {

// DOT_PRODUCT
template <typename T>
static Expr<T> FoldDotProduct(
    FoldingContext &context, FunctionRef<T> &&funcRef) {
  using Element = typename Constant<T>::Element;
  auto args{funcRef.arguments()};
  CHECK(args.size() == 2);
  Folder<T> folder{context};
  Constant<T> *va{folder.Folding(args[0])};
  Constant<T> *vb{folder.Folding(args[1])};
  if (va && vb) {
    CHECK(va->Rank() == 1 && vb->Rank() == 1);
    if (va->size() != vb->size()) {
      context.messages().Say(
          "Vector arguments to DOT_PRODUCT have distinct extents %zd and %zd"_err_en_US,
          va->size(), vb->size());
      return MakeInvalidIntrinsic(std::move(funcRef));
    }
    Element sum{};
    bool overflow{false};
    if constexpr (T::category == TypeCategory::Complex) {
      std::vector<Element> conjugates;
      for (const Element &x : va->values()) {
        conjugates.emplace_back(x.CONJG());
      }
      Constant<T> conjgA{
          std::move(conjugates), ConstantSubscripts{va->shape()}};
      Expr<T> products{Fold(
          context, Expr<T>{std::move(conjgA)} * Expr<T>{Constant<T>{*vb}})};
      Constant<T> &cProducts{DEREF(UnwrapConstantValue<T>(products))};
      [[maybe_unused]] Element correction{};
      const auto &rounding{context.targetCharacteristics().roundingMode()};
      for (const Element &x : cProducts.values()) {
        if constexpr (useKahanSummation) {
          auto next{correction.Add(x, rounding)};
          overflow |= next.flags.test(RealFlag::Overflow);
          auto added{sum.Add(next.value, rounding)};
          overflow |= added.flags.test(RealFlag::Overflow);
          correction = added.value.Subtract(sum, rounding)
                           .value.Subtract(next.value, rounding)
                           .value;
          sum = std::move(added.value);
        } else {
          auto added{sum.Add(x, rounding)};
          overflow |= added.flags.test(RealFlag::Overflow);
          sum = std::move(added.value);
        }
      }
    } else if constexpr (T::category == TypeCategory::Logical) {
      Expr<T> conjunctions{Fold(context,
          Expr<T>{LogicalOperation<T::kind>{LogicalOperator::And,
              Expr<T>{Constant<T>{*va}}, Expr<T>{Constant<T>{*vb}}}})};
      Constant<T> &cConjunctions{DEREF(UnwrapConstantValue<T>(conjunctions))};
      for (const Element &x : cConjunctions.values()) {
        if (x.IsTrue()) {
          sum = Element{true};
          break;
        }
      }
    } else if constexpr (T::category == TypeCategory::Integer) {
      Expr<T> products{
          Fold(context, Expr<T>{Constant<T>{*va}} * Expr<T>{Constant<T>{*vb}})};
      Constant<T> &cProducts{DEREF(UnwrapConstantValue<T>(products))};
      for (const Element &x : cProducts.values()) {
        auto next{sum.AddSigned(x)};
        overflow |= next.overflow;
        sum = std::move(next.value);
      }
    } else {
      static_assert(T::category == TypeCategory::Real);
      Expr<T> products{
          Fold(context, Expr<T>{Constant<T>{*va}} * Expr<T>{Constant<T>{*vb}})};
      Constant<T> &cProducts{DEREF(UnwrapConstantValue<T>(products))};
      [[maybe_unused]] Element correction{};
      const auto &rounding{context.targetCharacteristics().roundingMode()};
      for (const Element &x : cProducts.values()) {
        if constexpr (useKahanSummation) {
          auto next{correction.Add(x, rounding)};
          overflow |= next.flags.test(RealFlag::Overflow);
          auto added{sum.Add(next.value, rounding)};
          overflow |= added.flags.test(RealFlag::Overflow);
          correction = added.value.Subtract(sum, rounding)
                           .value.Subtract(next.value, rounding)
                           .value;
          sum = std::move(added.value);
        } else {
          auto added{sum.Add(x, rounding)};
          overflow |= added.flags.test(RealFlag::Overflow);
          sum = std::move(added.value);
        }
      }
    }
    if (overflow &&
        context.languageFeatures().ShouldWarn(
            common::UsageWarning::FoldingException)) {
      context.messages().Say(common::UsageWarning::FoldingException,
          "DOT_PRODUCT of %s data overflowed during computation"_warn_en_US,
          T::AsFortran());
    }
    return Expr<T>{Constant<T>{std::move(sum)}};
  }
  return Expr<T>{std::move(funcRef)};
}

// Fold and validate a DIM= argument.  Returns false on error.
bool CheckReductionDIM(std::optional<int> &dim, FoldingContext &,
    ActualArguments &, std::optional<int> dimIndex, int rank);

// Fold and validate a MASK= argument.  Return null on error, absent MASK=, or
// non-constant MASK=.
Constant<LogicalResult> *GetReductionMASK(
    std::optional<ActualArgument> &maskArg, const ConstantSubscripts &shape,
    FoldingContext &);

// Common preprocessing for reduction transformational intrinsic function
// folding.  If the intrinsic can have DIM= &/or MASK= arguments, extract
// and check them.  If a MASK= is present, apply it to the array data and
// substitute replacement values for elements corresponding to .FALSE. in
// the mask.  If the result is present, the intrinsic call can be folded.
template <typename T> struct ArrayAndMask {
  Constant<T> array;
  Constant<LogicalResult> mask;
};
template <typename T>
static std::optional<ArrayAndMask<T>> ProcessReductionArgs(
    FoldingContext &context, ActualArguments &arg, std::optional<int> &dim,
    int arrayIndex, std::optional<int> dimIndex = std::nullopt,
    std::optional<int> maskIndex = std::nullopt) {
  if (arg.empty()) {
    return std::nullopt;
  }
  Constant<T> *folded{Folder<T>{context}.Folding(arg[arrayIndex])};
  if (!folded || folded->Rank() < 1) {
    return std::nullopt;
  }
  if (!CheckReductionDIM(dim, context, arg, dimIndex, folded->Rank())) {
    return std::nullopt;
  }
  std::size_t n{folded->size()};
  std::vector<Scalar<LogicalResult>> maskElement;
  if (maskIndex && static_cast<std::size_t>(*maskIndex) < arg.size() &&
      arg[*maskIndex]) {
    if (const Constant<LogicalResult> *origMask{
            GetReductionMASK(arg[*maskIndex], folded->shape(), context)}) {
      if (auto scalarMask{origMask->GetScalarValue()}) {
        maskElement =
            std::vector<Scalar<LogicalResult>>(n, scalarMask->IsTrue());
      } else {
        maskElement = origMask->values();
      }
    } else {
      return std::nullopt;
    }
  } else {
    maskElement = std::vector<Scalar<LogicalResult>>(n, true);
  }
  return ArrayAndMask<T>{Constant<T>(*folded),
      Constant<LogicalResult>{
          std::move(maskElement), ConstantSubscripts{folded->shape()}}};
}

// Generalized reduction to an array of one dimension fewer (w/ DIM=)
// or to a scalar (w/o DIM=).  The ACCUMULATOR type must define
// operator()(Scalar<T> &, const ConstantSubscripts &, bool first)
// and Done(Scalar<T> &).
template <typename T, typename ACCUMULATOR, typename ARRAY>
static Constant<T> DoReduction(const Constant<ARRAY> &array,
    const Constant<LogicalResult> &mask, std::optional<int> &dim,
    const Scalar<T> &identity, ACCUMULATOR &accumulator) {
  ConstantSubscripts at{array.lbounds()};
  ConstantSubscripts maskAt{mask.lbounds()};
  std::vector<typename Constant<T>::Element> elements;
  ConstantSubscripts resultShape; // empty -> scalar
  if (dim) { // DIM= is present, so result is an array
    resultShape = array.shape();
    resultShape.erase(resultShape.begin() + (*dim - 1));
    ConstantSubscript dimExtent{array.shape().at(*dim - 1)};
    CHECK(dimExtent == mask.shape().at(*dim - 1));
    ConstantSubscript &dimAt{at[*dim - 1]};
    ConstantSubscript dimLbound{dimAt};
    ConstantSubscript &maskDimAt{maskAt[*dim - 1]};
    ConstantSubscript maskDimLbound{maskDimAt};
    for (auto n{GetSize(resultShape)}; n-- > 0;
         array.IncrementSubscripts(at), mask.IncrementSubscripts(maskAt)) {
      elements.push_back(identity);
      if (dimExtent > 0) {
        dimAt = dimLbound;
        maskDimAt = maskDimLbound;
        bool firstUnmasked{true};
        for (ConstantSubscript j{0}; j < dimExtent; ++j, ++dimAt, ++maskDimAt) {
          if (mask.At(maskAt).IsTrue()) {
            accumulator(elements.back(), at, firstUnmasked);
            firstUnmasked = false;
          }
        }
        --dimAt, --maskDimAt;
      }
      accumulator.Done(elements.back());
    }
  } else { // no DIM=, result is scalar
    elements.push_back(identity);
    bool firstUnmasked{true};
    for (auto n{array.size()}; n-- > 0;
         array.IncrementSubscripts(at), mask.IncrementSubscripts(maskAt)) {
      if (mask.At(maskAt).IsTrue()) {
        accumulator(elements.back(), at, firstUnmasked);
        firstUnmasked = false;
      }
    }
    accumulator.Done(elements.back());
  }
  if constexpr (T::category == TypeCategory::Character) {
    return {static_cast<ConstantSubscript>(identity.size()),
        std::move(elements), std::move(resultShape)};
  } else {
    return {std::move(elements), std::move(resultShape)};
  }
}

// MAXVAL & MINVAL
template <typename T, bool ABS = false> class MaxvalMinvalAccumulator {
public:
  MaxvalMinvalAccumulator(
      RelationalOperator opr, FoldingContext &context, const Constant<T> &array)
      : opr_{opr}, context_{context}, array_{array} {};
  void operator()(Scalar<T> &element, const ConstantSubscripts &at,
      [[maybe_unused]] bool firstUnmasked) const {
    auto aAt{array_.At(at)};
    if constexpr (ABS) {
      aAt = aAt.ABS();
    }
    if constexpr (T::category == TypeCategory::Real) {
      if (firstUnmasked || element.IsNotANumber()) {
        // Return NaN if and only if all unmasked elements are NaNs and
        // at least one unmasked element is visible.
        element = aAt;
        return;
      }
    }
    Expr<LogicalResult> test{PackageRelation(
        opr_, Expr<T>{Constant<T>{aAt}}, Expr<T>{Constant<T>{element}})};
    auto folded{GetScalarConstantValue<LogicalResult>(
        test.Rewrite(context_, std::move(test)))};
    CHECK(folded.has_value());
    if (folded->IsTrue()) {
      element = aAt;
    }
  }
  void Done(Scalar<T> &) const {}

private:
  RelationalOperator opr_;
  FoldingContext &context_;
  const Constant<T> &array_;
};

template <typename T>
static Expr<T> FoldMaxvalMinval(FoldingContext &context, FunctionRef<T> &&ref,
    RelationalOperator opr, const Scalar<T> &identity) {
  static_assert(T::category == TypeCategory::Integer ||
      T::category == TypeCategory::Real ||
      T::category == TypeCategory::Character);
  std::optional<int> dim;
  if (std::optional<ArrayAndMask<T>> arrayAndMask{
          ProcessReductionArgs<T>(context, ref.arguments(), dim,
              /*ARRAY=*/0, /*DIM=*/1, /*MASK=*/2)}) {
    MaxvalMinvalAccumulator accumulator{opr, context, arrayAndMask->array};
    return Expr<T>{DoReduction<T>(
        arrayAndMask->array, arrayAndMask->mask, dim, identity, accumulator)};
  }
  return Expr<T>{std::move(ref)};
}

// PRODUCT
template <typename T> class ProductAccumulator {
public:
  ProductAccumulator(const Constant<T> &array) : array_{array} {}
  void operator()(
      Scalar<T> &element, const ConstantSubscripts &at, bool /*first*/) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto prod{element.MultiplySigned(array_.At(at))};
      overflow_ |= prod.SignedMultiplicationOverflowed();
      element = prod.lower;
    } else { // Real & Complex
      auto prod{element.Multiply(array_.At(at))};
      overflow_ |= prod.flags.test(RealFlag::Overflow);
      element = prod.value;
    }
  }
  bool overflow() const { return overflow_; }
  void Done(Scalar<T> &) const {}

private:
  const Constant<T> &array_;
  bool overflow_{false};
};

template <typename T>
static Expr<T> FoldProduct(
    FoldingContext &context, FunctionRef<T> &&ref, Scalar<T> identity) {
  static_assert(T::category == TypeCategory::Integer ||
      T::category == TypeCategory::Real ||
      T::category == TypeCategory::Complex);
  std::optional<int> dim;
  if (std::optional<ArrayAndMask<T>> arrayAndMask{
          ProcessReductionArgs<T>(context, ref.arguments(), dim,
              /*ARRAY=*/0, /*DIM=*/1, /*MASK=*/2)}) {
    ProductAccumulator accumulator{arrayAndMask->array};
    auto result{Expr<T>{DoReduction<T>(
        arrayAndMask->array, arrayAndMask->mask, dim, identity, accumulator)}};
    if (accumulator.overflow() &&
        context.languageFeatures().ShouldWarn(
            common::UsageWarning::FoldingException)) {
      context.messages().Say(common::UsageWarning::FoldingException,
          "PRODUCT() of %s data overflowed"_warn_en_US, T::AsFortran());
    }
    return result;
  }
  return Expr<T>{std::move(ref)};
}

// SUM
template <typename T> class SumAccumulator {
  using Element = typename Constant<T>::Element;

public:
  SumAccumulator(const Constant<T> &array, Rounding rounding)
      : array_{array}, rounding_{rounding} {}
  void operator()(
      Element &element, const ConstantSubscripts &at, bool /*first*/) {
    if constexpr (T::category == TypeCategory::Integer) {
      auto sum{element.AddSigned(array_.At(at))};
      overflow_ |= sum.overflow;
      element = sum.value;
    } else { // Real & Complex: use Kahan summation
      auto next{array_.At(at).Add(correction_, rounding_)};
      overflow_ |= next.flags.test(RealFlag::Overflow);
      auto sum{element.Add(next.value, rounding_)};
      overflow_ |= sum.flags.test(RealFlag::Overflow);
      // correction = (sum - element) - next; algebraically zero
      correction_ = sum.value.Subtract(element, rounding_)
                        .value.Subtract(next.value, rounding_)
                        .value;
      element = sum.value;
    }
  }
  bool overflow() const { return overflow_; }
  void Done([[maybe_unused]] Element &element) {
    if constexpr (T::category != TypeCategory::Integer) {
      auto corrected{element.Add(correction_, rounding_)};
      overflow_ |= corrected.flags.test(RealFlag::Overflow);
      correction_ = Scalar<T>{};
      element = corrected.value;
    }
  }

private:
  const Constant<T> &array_;
  Rounding rounding_;
  bool overflow_{false};
  Element correction_{};
};

template <typename T>
static Expr<T> FoldSum(FoldingContext &context, FunctionRef<T> &&ref) {
  static_assert(T::category == TypeCategory::Integer ||
      T::category == TypeCategory::Real ||
      T::category == TypeCategory::Complex);
  using Element = typename Constant<T>::Element;
  std::optional<int> dim;
  Element identity{};
  if (std::optional<ArrayAndMask<T>> arrayAndMask{
          ProcessReductionArgs<T>(context, ref.arguments(), dim,
              /*ARRAY=*/0, /*DIM=*/1, /*MASK=*/2)}) {
    SumAccumulator accumulator{
        arrayAndMask->array, context.targetCharacteristics().roundingMode()};
    auto result{Expr<T>{DoReduction<T>(
        arrayAndMask->array, arrayAndMask->mask, dim, identity, accumulator)}};
    if (accumulator.overflow() &&
        context.languageFeatures().ShouldWarn(
            common::UsageWarning::FoldingException)) {
      context.messages().Say(common::UsageWarning::FoldingException,
          "SUM() of %s data overflowed"_warn_en_US, T::AsFortran());
    }
    return result;
  }
  return Expr<T>{std::move(ref)};
}

// Utility for IALL, IANY, IPARITY, ALL, ANY, & PARITY
template <typename T> class OperationAccumulator {
public:
  OperationAccumulator(const Constant<T> &array,
      Scalar<T> (Scalar<T>::*operation)(const Scalar<T> &) const)
      : array_{array}, operation_{operation} {}
  void operator()(
      Scalar<T> &element, const ConstantSubscripts &at, bool /*first*/) {
    element = (element.*operation_)(array_.At(at));
  }
  void Done(Scalar<T> &) const {}

private:
  const Constant<T> &array_;
  Scalar<T> (Scalar<T>::*operation_)(const Scalar<T> &) const;
};

} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_FOLD_REDUCTION_H_
