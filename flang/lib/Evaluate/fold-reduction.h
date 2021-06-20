//===-- lib/Evaluate/fold-reduction.h -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// TODO: ALL, ANY, COUNT, DOT_PRODUCT, FINDLOC, IALL, IANY, IPARITY,
// NORM2, MAXLOC, MINLOC, PARITY, PRODUCT, SUM

#ifndef FORTRAN_EVALUATE_FOLD_REDUCTION_H_
#define FORTRAN_EVALUATE_FOLD_REDUCTION_H_

#include "fold-implementation.h"

namespace Fortran::evaluate {

// MAXVAL & MINVAL
template <typename T>
Expr<T> FoldMaxvalMinval(FoldingContext &context, FunctionRef<T> &&ref,
    RelationalOperator opr, Scalar<T> identity) {
  static_assert(T::category == TypeCategory::Integer ||
      T::category == TypeCategory::Real ||
      T::category == TypeCategory::Character);
  using Element = typename Constant<T>::Element;
  auto &arg{ref.arguments()};
  CHECK(arg.size() <= 3);
  if (arg.empty()) {
    return Expr<T>{std::move(ref)};
  }
  Constant<T> *array{Folder<T>{context}.Folding(arg[0])};
  if (!array || array->Rank() < 1) {
    return Expr<T>{std::move(ref)};
  }
  std::optional<ConstantSubscript> dim;
  if (arg.size() >= 2 && arg[1]) {
    if (auto *dimConst{Folder<SubscriptInteger>{context}.Folding(arg[1])}) {
      if (auto dimScalar{dimConst->GetScalarValue()}) {
        dim.emplace(dimScalar->ToInt64());
        if (*dim < 1 || *dim > array->Rank()) {
          context.messages().Say(
              "DIM=%jd is not valid for an array of rank %d"_err_en_US,
              static_cast<std::intmax_t>(*dim), array->Rank());
          dim.reset();
        }
      }
    }
    if (!dim) {
      return Expr<T>{std::move(ref)};
    }
  }
  Constant<LogicalResult> *mask{};
  if (arg.size() >= 3 && arg[2]) {
    mask = Folder<LogicalResult>{context}.Folding(arg[2]);
    if (!mask) {
      return Expr<T>{std::move(ref)};
    }
    if (!CheckConformance(context.messages(), AsShape(array->shape()),
            AsShape(mask->shape()),
            CheckConformanceFlags::RightScalarExpandable, "ARRAY=", "MASK=")
             .value_or(false)) {
      return Expr<T>{std::move(ref)};
    }
  }
  // Do it
  ConstantSubscripts at{array->lbounds()}, maskAt;
  bool maskAllFalse{false};
  if (mask) {
    if (auto scalar{mask->GetScalarValue()}) {
      if (scalar->IsTrue()) {
        mask = nullptr; // all .TRUE.
      } else {
        maskAllFalse = true;
      }
    } else {
      maskAt = mask->lbounds();
    }
  }
  std::vector<Element> result;
  ConstantSubscripts resultShape; // empty -> scalar
  // Internal function to accumulate into result.back().
  auto Accumulate{[&]() {
    if (!maskAllFalse && (maskAt.empty() || mask->At(maskAt).IsTrue())) {
      Expr<LogicalResult> test{
          PackageRelation(opr, Expr<T>{Constant<T>{array->At(at)}},
              Expr<T>{Constant<T>{result.back()}})};
      auto folded{GetScalarConstantValue<LogicalResult>(
          test.Rewrite(context, std::move(test)))};
      CHECK(folded.has_value());
      if (folded->IsTrue()) {
        result.back() = array->At(at);
      }
    }
  }};
  if (dim) { // DIM= is present, so result is an array
    resultShape = array->shape();
    resultShape.erase(resultShape.begin() + (*dim - 1));
    ConstantSubscript dimExtent{array->shape().at(*dim - 1)};
    ConstantSubscript &dimAt{at[*dim - 1]};
    ConstantSubscript dimLbound{dimAt};
    ConstantSubscript *maskDimAt{maskAt.empty() ? nullptr : &maskAt[*dim - 1]};
    ConstantSubscript maskLbound{maskDimAt ? *maskDimAt : 0};
    for (auto n{GetSize(resultShape)}; n-- > 0;
         IncrementSubscripts(at, array->shape())) {
      dimAt = dimLbound;
      if (maskDimAt) {
        *maskDimAt = maskLbound;
      }
      result.push_back(identity);
      for (ConstantSubscript j{0}; j < dimExtent;
           ++j, ++dimAt, maskDimAt && ++*maskDimAt) {
        Accumulate();
      }
      if (maskDimAt) {
        IncrementSubscripts(maskAt, mask->shape());
      }
    }
  } else { // no DIM=, result is scalar
    result.push_back(identity);
    for (auto n{array->size()}; n-- > 0;
         IncrementSubscripts(at, array->shape())) {
      Accumulate();
      if (!maskAt.empty()) {
        IncrementSubscripts(maskAt, mask->shape());
      }
    }
  }
  if constexpr (T::category == TypeCategory::Character) {
    return Expr<T>{Constant<T>{static_cast<ConstantSubscript>(identity.size()),
        std::move(result), std::move(resultShape)}};
  } else {
    return Expr<T>{Constant<T>{std::move(result), std::move(resultShape)}};
  }
}

} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_FOLD_REDUCTION_H_
