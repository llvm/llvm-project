//===-- Implementation of fegetexceptflag function ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/fenv/fegetexceptflag.h"
#include "hdr/types/fexcept_t.h"
#include "src/__support/FPUtil/FEnvImpl.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, fegetexceptflag, (fexcept_t * flagp, int excepts)) {
  static_assert(sizeof(int) >= sizeof(fexcept_t),
                "fexcept_t value cannot fit in an int value.");
  *flagp = static_cast<fexcept_t>(fputil::test_except(FE_ALL_EXCEPT) & excepts);
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
