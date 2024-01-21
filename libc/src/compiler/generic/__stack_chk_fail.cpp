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

extern "C" {

void __stack_chk_fail(void) {
  LIBC_NAMESPACE::write_to_stderr("stack smashing detected");
  LIBC_NAMESPACE::abort();
}

} // extern "C"
