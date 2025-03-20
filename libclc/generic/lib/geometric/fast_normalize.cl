//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc.h>

_CLC_OVERLOAD _CLC_DEF float fast_normalize(float p) {
  return normalize(p);
}

#define __CLC_BODY <fast_normalize.inc>
#define __FLOAT_ONLY
#include <clc/geometric/floatn.inc>
#undef __FLOAT_ONLY
