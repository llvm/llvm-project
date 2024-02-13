//===-- Implementation of feenableexcept function -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/feenableexcept.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, feenableexcept, (int e)) {
  return fputil::enable_except(e);
}

} // namespace LIBC_NAMESPACE
