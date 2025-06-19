//===-- Implementation for memalignment -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/memalignment.h"
#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(size_t, memalignment, (const void *p)) {
  if (p == nullptr)
    return 0;

  uintptr_t addr = reinterpret_cast<uintptr_t>(p);

  return size_t(1) << cpp::countr_zero(addr);
}

} // namespace LIBC_NAMESPACE_DECL
