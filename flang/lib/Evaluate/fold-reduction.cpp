//===-- lib/Evaluate/fold-reduction.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "fold-reduction.h"

namespace Fortran::evaluate {
bool CheckReductionDIM(std::optional<int> &dim, FoldingContext &context,
    ActualArguments &arg, std::optional<int> dimIndex, int rank) {
  if (!dimIndex || static_cast<std::size_t>(*dimIndex) >= arg.size() ||
      !arg[*dimIndex]) {
    dim.reset();
    return true; // no DIM= argument
  }
  if (auto *dimConst{
          Folder<SubscriptInteger>{context}.Folding(arg[*dimIndex])}) {
    if (auto dimScalar{dimConst->GetScalarValue()}) {
      auto dimVal{dimScalar->ToInt64()};
      if (dimVal >= 1 && dimVal <= rank) {
        dim = dimVal;
        return true; // DIM= exists and is a valid constant
      } else {
        context.messages().Say(
            "DIM=%jd is not valid for an array of rank %d"_err_en_US,
            static_cast<std::intmax_t>(dimVal), rank);
      }
    }
  }
  return false; // DIM= bad or not scalar constant
}

Constant<LogicalResult> *GetReductionMASK(
    std::optional<ActualArgument> &maskArg, const ConstantSubscripts &shape,
    FoldingContext &context) {
  Constant<LogicalResult> *mask{
      Folder<LogicalResult>{context}.Folding(maskArg)};
  if (mask &&
      !CheckConformance(context.messages(), AsShape(shape),
          AsShape(mask->shape()), CheckConformanceFlags::RightScalarExpandable,
          "ARRAY=", "MASK=")
           .value_or(false)) {
    mask = nullptr;
  }
  return mask;
}
} // namespace Fortran::evaluate
