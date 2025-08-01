//===-- asan_win_dynamic_runtime_thunk.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// This file defines things that need to be present for application modules
// that are dynamic linked with the C Runtime.
//
//===----------------------------------------------------------------------===//

#ifdef SANITIZER_DYNAMIC_RUNTIME_THUNK
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>

#  include "asan_win_common_runtime_thunk.h"
#  include "sanitizer_common/sanitizer_win_defs.h"

////////////////////////////////////////////////////////////////////////////////
// For some reason, the MD CRT doesn't call the C/C++ terminators during on DLL
// unload or on exit.  ASan relies on LLVM global_dtors to call
// __asan_unregister_globals on these events, which unfortunately doesn't work
// with the MD runtime, see PR22545 for the details.
// To work around this, for each DLL we schedule a call to UnregisterGlobals
// using atexit() that calls a small subset of C terminators
// where LLVM global_dtors is placed.  Fingers crossed, no other C terminators
// are there.
extern "C" int __cdecl atexit(void(__cdecl *f)(void));
extern "C" void __cdecl _initterm(void *a, void *b);

namespace {
__declspec(allocate(".CRT$XTW")) void *before_global_dtors = 0;
__declspec(allocate(".CRT$XTY")) void *after_global_dtors = 0;

void UnregisterGlobals() {
  _initterm(&before_global_dtors, &after_global_dtors);
}

int ScheduleUnregisterGlobals() { return atexit(UnregisterGlobals); }
}  // namespace

// We need to call 'atexit(UnregisterGlobals);' as early as possible, but after
// atexit() is initialized (.CRT$XIC).  As this is executed before C++
// initializers (think ctors for globals), UnregisterGlobals gets executed after
// dtors for C++ globals.
extern "C" __declspec(allocate(".CRT$XID")) int (
    *__asan_schedule_unregister_globals)() = ScheduleUnregisterGlobals;
WIN_FORCE_LINK(__asan_schedule_unregister_globals)

#endif  // SANITIZER_DYNAMIC_RUNTIME_THUNK
