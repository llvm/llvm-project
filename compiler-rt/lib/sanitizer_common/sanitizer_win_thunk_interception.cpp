//===-- sanitizer_win_thunk_interception.cpp -----------------------  -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines things that need to be present in the application modules
// to interact with sanitizer DLL correctly and cannot be implemented using the
// default "import library" generated when linking the DLL.
//
// This includes the common infrastructure required to intercept local functions
// that must be replaced with sanitizer-aware versions, as well as the
// registration of weak functions with the sanitizer DLL. With this in-place,
// other sanitizer components can simply write to the .INTR and .WEAK sections.
//
//===----------------------------------------------------------------------===//

#if defined(SANITIZER_STATIC_RUNTIME_THUNK) || \
    defined(SANITIZER_DYNAMIC_RUNTIME_THUNK)
#  include "sanitizer_win_thunk_interception.h"

extern "C" void abort();

namespace __sanitizer {

int override_function(const char *export_name, const uptr user_function) {
  if (!__sanitizer_override_function(export_name, user_function)) {
    abort();
  }

  return 0;
}

int register_weak(const char *export_name, const uptr user_function) {
  if (!__sanitizer_register_weak_function(export_name, user_function)) {
    abort();
  }

  return 0;
}

void initialize_thunks(const sanitizer_thunk *first,
                       const sanitizer_thunk *last) {
  for (const sanitizer_thunk *it = first; it < last; ++it) {
    if (*it) {
      (*it)();
    }
  }
}
}  // namespace __sanitizer

#  define INTERFACE_FUNCTION(Name)
#  define INTERFACE_WEAK_FUNCTION(Name) REGISTER_WEAK_FUNCTION(Name)
#  include "sanitizer_common_interface.inc"

#  pragma section(".INTR$A", read)  // intercept begin
#  pragma section(".INTR$Z", read)  // intercept end
#  pragma section(".WEAK$A", read)  // weak begin
#  pragma section(".WEAK$Z", read)  // weak end

extern "C" {
__declspec(allocate(
    ".INTR$A")) sanitizer_thunk __sanitizer_intercept_thunk_begin;
__declspec(allocate(".INTR$Z")) sanitizer_thunk __sanitizer_intercept_thunk_end;

__declspec(allocate(
    ".WEAK$A")) sanitizer_thunk __sanitizer_register_weak_thunk_begin;
__declspec(allocate(
    ".WEAK$Z")) sanitizer_thunk __sanitizer_register_weak_thunk_end;
}

extern "C" int __sanitizer_thunk_init() {
  // __sanitizer_static_thunk_init is expected to be called by only one thread.
  static bool flag = false;
  if (flag) {
    return 0;
  }
  flag = true;

  __sanitizer::initialize_thunks(&__sanitizer_intercept_thunk_begin,
                                 &__sanitizer_intercept_thunk_end);
  __sanitizer::initialize_thunks(&__sanitizer_register_weak_thunk_begin,
                                 &__sanitizer_register_weak_thunk_end);

  // In DLLs, the callbacks are expected to return 0,
  // otherwise CRT initialization fails.
  return 0;
}

// We want to call dll_thunk_init before C/C++ initializers / constructors are
// executed, otherwise functions like memset might be invoked.
#  pragma section(".CRT$XIB", long, read)
__declspec(allocate(".CRT$XIB")) int (*__sanitizer_thunk_init_ptr)() =
    __sanitizer_thunk_init;

static void WINAPI sanitizer_thunk_thread_init(void *mod, unsigned long reason,
                                               void *reserved) {
  if (reason == /*DLL_PROCESS_ATTACH=*/1)
    __sanitizer_thunk_init();
}

#  pragma section(".CRT$XLAB", long, read)
__declspec(allocate(".CRT$XLAB")) void(
    WINAPI *__sanitizer_thunk_thread_init_ptr)(void *, unsigned long, void *) =
    sanitizer_thunk_thread_init;

#endif  // defined(SANITIZER_STATIC_RUNTIME_THUNK) ||
        // defined(SANITIZER_DYNAMIC_RUNTIME_THUNK)