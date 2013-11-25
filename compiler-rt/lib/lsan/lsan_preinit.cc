//===-- lsan_preinit.cc ---------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
//
// Call __lsan_init at the very early stage of process startup.
//===----------------------------------------------------------------------===//

#include "lsan.h"

#ifndef LSAN_USE_PREINIT_ARRAY
#define LSAN_USE_PREINIT_ARRAY 1
#endif

#if LSAN_USE_PREINIT_ARRAY && !defined(PIC)
  // We force __lsan_init to be called before anyone else by placing it into
  // .preinit_array section.
  __attribute__((section(".preinit_array"), used))
  void (*__local_lsan_preinit)(void) = __lsan_init;
#endif
