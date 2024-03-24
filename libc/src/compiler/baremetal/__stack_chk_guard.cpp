//===-- Implementation of __stack_chk_guard -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/compiler/__stack_chk_guard.h"

#include <stdint.h>

extern "C" {

uintptr_t __stack_chk_guard = 0x00000aff; // 0, 0, '\n', 255

} // extern "C"
