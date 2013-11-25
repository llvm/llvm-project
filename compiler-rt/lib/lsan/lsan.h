//=-- lsan.h --------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
// Private header for standalone LSan RTL.
//
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_flags.h"
#include "sanitizer_common/sanitizer_stacktrace.h"

namespace __lsan {

void InitializeInterceptors();

}  // namespace __lsan

extern bool lsan_inited;
extern bool lsan_init_is_running;

extern "C" void __lsan_init();
