//===-- lsan_testlib.cc ---------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
// Standalone LSan tool as a shared library, to be used with LD_PRELOAD.
//
//===----------------------------------------------------------------------===//
/* Usage:
clang++ ../sanitizer_common/sanitizer_*.cc ../interception/interception_*.cc \
 lsan*.cc tests/lsan_testlib.cc -I. -I.. -g -ldl -lpthread -fPIC -shared -O2 \
 -DLSAN_USE_PREINIT_ARRAY=0 -o lsan.so
LD_PRELOAD=./lsan.so /your/app
*/
#include "lsan.h"

__attribute__((constructor))
void constructor() {
  __lsan_init();
}
