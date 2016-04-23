//===-- esan.h --------------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of EfficiencySanitizer, a family of performance tuners.
//
// Main internal esan header file.
//
// Ground rules:
//   - C++ run-time should not be used (static CTORs, RTTI, exceptions, static
//     function-scope locals)
//   - All functions/classes/etc reside in namespace __esan, except for those
//     declared in esan_interface_internal.h.
//   - Platform-specific files should be used instead of ifdefs (*).
//   - No system headers included in header files (*).
//   - Platform specific headers included only into platform-specific files (*).
//
//  (*) Except when inlining is critical for performance.
//===----------------------------------------------------------------------===//

#ifndef ESAN_H
#define ESAN_H

#include "sanitizer_common/sanitizer_common.h"
#include "esan_interface_internal.h"

namespace __esan {

extern bool EsanIsInitialized;

extern ToolType WhichTool;

void initializeLibrary(ToolType Tool);
int finalizeLibrary();
void processRangeAccess(uptr PC, uptr Addr, int Size, bool IsWrite);
void initializeInterceptors();

} // namespace __esan

#endif // ESAN_H
