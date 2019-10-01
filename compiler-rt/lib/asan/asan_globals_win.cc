//===-- asan_globals_win.cc -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Global registration code that is linked into every Windows DLL and EXE.
//
//===----------------------------------------------------------------------===//

#include "asan_interface_internal.h"
#if SANITIZER_WINDOWS

namespace __asan {

#pragma section(".ASAN$GA", read, write)  // NOLINT
#pragma section(".ASAN$GZ", read, write)  // NOLINT
extern "C" __declspec(allocate(".ASAN$GA"))
    ALIGNED(sizeof(__asan_global)) __asan_global __asan_globals_start = {};
extern "C" __declspec(allocate(".ASAN$GZ"))
    ALIGNED(sizeof(__asan_global)) __asan_global __asan_globals_end = {};
#pragma comment(linker, "/merge:.ASAN=.data")

static void call_on_globals(void (*hook)(__asan_global *, uptr)) {
  __asan_global *start = &__asan_globals_start + 1;
  __asan_global *end = &__asan_globals_end;
  uptr bytediff = (uptr)end - (uptr)start;
  if (bytediff % sizeof(__asan_global) != 0) {
#if defined(SANITIZER_DLL_THUNK) || defined(SANITIZER_DYNAMIC_RUNTIME_THUNK)
    __debugbreak();
#else
    CHECK("corrupt asan global array");
#endif
  }
  // We know end >= start because the linker sorts the portion after the dollar
  // sign alphabetically.
  uptr n = end - start;
  hook(start, n);
}

static void register_dso_globals() {
  call_on_globals(&__asan_register_globals);
}

static void unregister_dso_globals() {
  call_on_globals(&__asan_unregister_globals);
}

// Register globals
#pragma section(".CRT$XCU", long, read)  // NOLINT
#pragma section(".CRT$XTX", long, read)  // NOLINT
extern "C" __declspec(allocate(".CRT$XCU"))
void (*const __asan_dso_reg_hook)() = &register_dso_globals;
extern "C" __declspec(allocate(".CRT$XTX"))
void (*const __asan_dso_unreg_hook)() = &unregister_dso_globals;

} // namespace __asan

#endif  // SANITIZER_WINDOWS
