//===-- Implementation of __stack_chk_guard -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/compiler/__stack_chk_guard.h"
#include "src/__support/OSUtil/syscall.h"
#include "src/errno/libc_errno.h"

#include <sys/syscall.h>

extern "C" {

uintptr_t __stack_chk_guard = 0;

} // extern "C"

namespace LIBC_NAMESPACE {
namespace {

class StackCheckGuard {
public:
  StackCheckGuard() {
    // TODO: Use getauxval(AT_RANDOM) once available.
    long retval =
        syscall_impl(SYS_getrandom, reinterpret_cast<long>(&__stack_chk_guard),
                     sizeof(uintptr_t), 0);
    if (retval < 0)
      __stack_chk_guard = 0x00000aff; // 0, 0, '\n', 255
  }
};

StackCheckGuard stack_check_guard;

} // anonymous namespace
} // namespace LIBC_NAMESPACE
