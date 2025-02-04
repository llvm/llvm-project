//===-- Implementation of __stack_chk_fail --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/compiler/__stack_chk_fail.h"
#include "src/__support/OSUtil/io.h"
#include "src/stdlib/abort.h"
#include <stdint.h> // For uintptr_t

extern "C" {

uintptr_t __stack_chk_guard = static_cast<uintptr_t>(0xa9fff01234);

void __stack_chk_fail(void) {
  LIBC_NAMESPACE::write_to_stderr("stack smashing detected\n");
  LIBC_NAMESPACE::abort();
}

} // extern "C"
