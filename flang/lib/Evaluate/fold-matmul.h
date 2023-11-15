//===-- lib/Evaluate/fold-matmul.h ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_FOLD_MATMUL_H_
#define FORTRAN_EVALUATE_FOLD_MATMUL_H_

#include "fold-implementation.h"

namespace Fortran::evaluate {

template <typename T>
static Expr<T> FoldMatmul(FoldingContext &context, FunctionRef<T> &&funcRef) {
  using Element = typename Constant<T>::Element;
  auto args{funcRef.arguments()};
  CHECK(args.size() == 2);
  Folder<T> folder{context};
  Constant<T> *ma{folder.Folding(args[0])};
  Constant<T> *mb{folder.Folding(args[1])};
  if (!ma || !mb) {
    return Expr<T>{std::move(funcRef)};
  }
  CHECK(ma->Rank() >= 1 && ma->Rank() <= 2 && mb->Rank() >= 1 &&
      mb->Rank() <= 2 && (ma->Rank() == 2 || mb->Rank() == 2));
  ConstantSubscript commonExtent{ma->shape().back()};
  if (mb->shape().front() != commonExtent) {
    context.messages().Say(
        "Arguments to MATMUL have distinct extents %zd and %zd on their last and first dimensions"_err_en_US,
        commonExtent, mb->shape().front());
    return MakeInvalidIntrinsic(std::move(funcRef));
  }
  ConstantSubscript rows{ma->Rank() == 1 ? 1 : ma->shape()[0]};
  ConstantSubscript columns{mb->Rank() == 1 ? 1 : mb->shape()[1]};
  std::vector<Element> elements;
  elements.reserve(rows * columns);
  bool overflow{false};
  [[maybe_unused]] const auto &rounding{
      context.targetCharacteristics().roundingMode()};
  // result(j,k) = SUM(A(j,:) * B(:,k))
  for (ConstantSubscript ci{0}; ci < columns; ++ci) {
    for (ConstantSubscript ri{0}; ri < rows; ++ri) {
      ConstantSubscripts aAt{ma->lbounds()};
      if (ma->Rank() == 2) {
        aAt[0] += ri;
      }
      ConstantSubscripts bAt{mb->lbounds()};
      if (mb->Rank() == 2) {
        bAt[1] += ci;
      }
      Element sum{};
      [[maybe_unused]] Element correction{};
      for (ConstantSubscript j{0}; j < commonExtent; ++j) {
        Element aElt{ma->At(aAt)};
        Element bElt{mb->At(bAt)};
        if constexpr (T::category == TypeCategory::Real ||
            T::category == TypeCategory::Complex) {
          // Kahan summation
          auto product{aElt.Multiply(bElt, rounding)};
          overflow |= product.flags.test(RealFlag::Overflow);
          auto next{correction.Add(product.value, rounding)};
          overflow |= next.flags.test(RealFlag::Overflow);
          auto added{sum.Add(next.value, rounding)};
          overflow |= added.flags.test(RealFlag::Overflow);
          correction = added.value.Subtract(sum, rounding)
                           .value.Subtract(next.value, rounding)
                           .value;
          sum = std::move(added.value);
        } else if constexpr (T::category == TypeCategory::Integer) {
          auto product{aElt.MultiplySigned(bElt)};
          overflow |= product.SignedMultiplicationOverflowed();
          auto added{sum.AddSigned(product.lower)};
          overflow |= added.overflow;
          sum = std::move(added.value);
        } else {
          static_assert(T::category == TypeCategory::Logical);
          sum = sum.OR(aElt.AND(bElt));
        }
        ++aAt.back();
        ++bAt.front();
      }
      elements.push_back(sum);
    }
  }
  if (overflow) {
    context.messages().Say(
        "MATMUL of %s data overflowed during computation"_warn_en_US,
        T::AsFortran());
  }
  ConstantSubscripts shape;
  if (ma->Rank() == 2) {
    shape.push_back(rows);
  }
  if (mb->Rank() == 2) {
    shape.push_back(columns);
  }
  return Expr<T>{Constant<T>{std::move(elements), std::move(shape)}};
}
} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_FOLD_MATMUL_H_
