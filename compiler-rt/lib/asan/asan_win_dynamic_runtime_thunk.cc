//===-- asan_win_uar_thunk.cc ---------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// This file defines things that need to be present in the application modules
// to interact with the ASan DLL runtime correctly and can't be implemented
// using the default "import library" generated when linking the DLL RTL.
//
// This includes:
//  - forwarding the detect_stack_use_after_return runtime option
//  - working around deficiencies of the MD runtime
//  - installing a custom SEH handlerx
//
//===----------------------------------------------------------------------===//

// Only compile this code when buidling asan_dynamic_runtime_thunk.lib
// Using #ifdef rather than relying on Makefiles etc.
// simplifies the build procedure.
#ifdef ASAN_DYNAMIC_RUNTIME_THUNK
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

// First, declare CRT sections we'll be using in this file
#pragma section(".CRT$XID", long, read)  // NOLINT
#pragma section(".CRT$XCAB", long, read)  // NOLINT
#pragma section(".CRT$XTW", long, read)  // NOLINT
#pragma section(".CRT$XTY", long, read)  // NOLINT

////////////////////////////////////////////////////////////////////////////////
// Define a copy of __asan_option_detect_stack_use_after_return that should be
// used when linking an MD runtime with a set of object files on Windows.
//
// The ASan MD runtime dllexports '__asan_option_detect_stack_use_after_return',
// so normally we would just dllimport it.  Unfortunately, the dllimport
// attribute adds __imp_ prefix to the symbol name of a variable.
// Since in general we don't know if a given TU is going to be used
// with a MT or MD runtime and we don't want to use ugly __imp_ names on Windows
// just to work around this issue, let's clone the a variable that is
// constant after initialization anyways.
extern "C" {
__declspec(dllimport) int __asan_should_detect_stack_use_after_return();
int __asan_option_detect_stack_use_after_return =
    __asan_should_detect_stack_use_after_return();
}

////////////////////////////////////////////////////////////////////////////////
// For some reason, the MD CRT doesn't call the C/C++ terminators during on DLL
// unload or on exit.  ASan relies on LLVM global_dtors to call
// __asan_unregister_globals on these events, which unfortunately doesn't work
// with the MD runtime, see PR22545 for the details.
// To work around this, for each DLL we schedule a call to UnregisterGlobals
// using atexit() that calls a small subset of C terminators
// where LLVM global_dtors is placed.  Fingers crossed, no other C terminators
// are there.
extern "C" int __cdecl atexit(void (__cdecl *f)(void));
extern "C" void __cdecl _initterm(void *a, void *b);

namespace {
__declspec(allocate(".CRT$XTW")) void* before_global_dtors = 0;
__declspec(allocate(".CRT$XTY")) void* after_global_dtors = 0;

void UnregisterGlobals() {
  _initterm(&before_global_dtors, &after_global_dtors);
}

int ScheduleUnregisterGlobals() {
  return atexit(UnregisterGlobals);
}

// We need to call 'atexit(UnregisterGlobals);' as early as possible, but after
// atexit() is initialized (.CRT$XIC).  As this is executed before C++
// initializers (think ctors for globals), UnregisterGlobals gets executed after
// dtors for C++ globals.
__declspec(allocate(".CRT$XID"))
int (*__asan_schedule_unregister_globals)() = ScheduleUnregisterGlobals;

}  // namespace

////////////////////////////////////////////////////////////////////////////////
// ASan SEH handling.
// We need to set the ASan-specific SEH handler at the end of CRT initialization
// of each module (see also asan_win.cc).
extern "C" {
__declspec(dllimport) int __asan_set_seh_filter();
static int SetSEHFilter() { return __asan_set_seh_filter(); }

// Unfortunately, putting a pointer to __asan_set_seh_filter into
// __asan_intercept_seh gets optimized out, so we have to use an extra function.
__declspec(allocate(".CRT$XCAB")) int (*__asan_seh_interceptor)() =
    SetSEHFilter;
}

#endif // ASAN_DYNAMIC_RUNTIME_THUNK
