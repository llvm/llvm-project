//===-- sanitizer_coverage_win_sections.cc --------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines delimiters for Sanitizer Coverage's section.
//===----------------------------------------------------------------------===//

#include "sanitizer_platform.h"
#if SANITIZER_WINDOWS
#include <stdint.h>
#pragma section(".SCOV$A", read, write)  // NOLINT
#pragma section(".SCOV$Z", read, write)  // NOLINT
extern "C" {
__declspec(allocate(".SCOV$A")) uint32_t __start___sancov_guards = 0;
__declspec(allocate(".SCOV$Z")) uint32_t __stop___sancov_guards = 0;
}
#endif // SANITIZER_WINDOWS
