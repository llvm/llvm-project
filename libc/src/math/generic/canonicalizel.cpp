//===-- Implementation of canonicalizel function
//----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------------------------------------===//

#include "src/math/canonicalizel.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/FPUtil/FPBits.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, canonicalizel,
                   (long double *cx, const long double *x)) {
  using FPB = fputil::FPBits<long double>;
  FPB sx(*x);
  if (sx.is_signaling_nan())
    fputil::raise_except_if_required(FE_INVALID);
  *cx = *x;
  return 0;
}

} // namespace LIBC_NAMESPACE
