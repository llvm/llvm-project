//===-- lsan_preinit.cc ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
//
// Call __lsan_init at the very early stage of process startup.
//===----------------------------------------------------------------------===//

#include "lsan.h"

#if SANITIZER_CAN_USE_PREINIT_ARRAY
  // We force __lsan_init to be called before anyone else by placing it into
  // .preinit_array section.
  __attribute__((section(".preinit_array"), used))
  void (*__local_lsan_preinit)(void) = __lsan_init;
#endif
