//===-- lib/runtime/extrema-minloc-dim.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// MINLOC with DIM= entry point.

#include "extrema.h"

namespace Fortran::runtime {
extern "C" {
RT_EXT_API_GROUP_BEGIN

void RTDEF(MinlocDim)(Descriptor &result, const Descriptor &x, int kind,
    int dim, const char *source, int line, const Descriptor *mask, bool back) {
  TypedPartialMaxOrMinLoc<false>(
      "MINLOC", result, x, kind, dim, source, line, mask, back);
}

RT_EXT_API_GROUP_END
} // extern "C"
} // namespace Fortran::runtime
