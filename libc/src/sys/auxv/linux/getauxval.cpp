//===-- Implementation file for getauxval function --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/auxv/getauxval.h"
#include "src/__support/OSUtil/linux/auxv.h"
#include "src/__support/libc_errno.h"

namespace LIBC_NAMESPACE_DECL {
LLVM_LIBC_FUNCTION(unsigned long, getauxval, (unsigned long id)) {
  if (cpp::optional<unsigned long> val = auxv::get(id))
    return *val;
  libc_errno = ENOENT;
  return 0;
}
} // namespace LIBC_NAMESPACE_DECL
