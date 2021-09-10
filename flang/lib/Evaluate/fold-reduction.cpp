//===-- lib/Evaluate/fold-reduction.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "fold-reduction.h"

namespace Fortran::evaluate {

std::optional<ConstantSubscript> CheckDIM(
    FoldingContext &context, std::optional<ActualArgument> &arg, int rank) {
  if (arg) {
    if (auto *dimConst{Folder<SubscriptInteger>{context}.Folding(arg)}) {
      if (auto dimScalar{dimConst->GetScalarValue()}) {
        auto dim{dimScalar->ToInt64()};
        if (dim >= 1 && dim <= rank) {
          return {dim};
        } else {
          context.messages().Say(
              "DIM=%jd is not valid for an array of rank %d"_err_en_US,
              static_cast<std::intmax_t>(dim), rank);
        }
      }
    }
  }
  return std::nullopt;
}

} // namespace Fortran::evaluate
