//===-- asan_win_common_runtime_thunk.cpp --------------------------- -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// This file defines things that need to be present in the application modules
// to interact with the ASan DLL runtime correctly and can't be implemented
// using the default "import library" generated when linking the DLL.
//
// This includes:
//  - Cloning shadow memory dynamic address from ASAN DLL
//  - Creating weak aliases to default implementation imported from asan dll
//  - Forwarding the detect_stack_use_after_return runtime option
//  - installing a custom SEH handler
//
//===----------------------------------------------------------------------===//

#if defined(SANITIZER_DYNAMIC_RUNTIME_THUNK) || \
    defined(SANITIZER_STATIC_RUNTIME_THUNK)
#  define SANITIZER_IMPORT_INTERFACE 1
#  define WIN32_LEAN_AND_MEAN
#  include "asan_win_common_runtime_thunk.h"

#  include <windows.h>

#  include "sanitizer_common/sanitizer_win_defs.h"
#  include "sanitizer_common/sanitizer_win_thunk_interception.h"

// Define weak alias for all weak functions imported from asan dll.
#  define INTERFACE_FUNCTION(Name)
#  define INTERFACE_WEAK_FUNCTION(Name) REGISTER_WEAK_FUNCTION(Name)
#  include "asan_interface.inc"

////////////////////////////////////////////////////////////////////////////////
// Define a copy of __asan_option_detect_stack_use_after_return that should be
// used when linking an MD runtime with a set of object files on Windows.
//
// The ASan MD runtime dllexports '__asan_option_detect_stack_use_after_return',
// so normally we would just dllimport it.  Unfortunately, the dllimport
// attribute adds __imp_ prefix to the symbol name of a variable.
// Since in general we don't know if a given TU is going to be used
// with a MT or MD runtime and we don't want to use ugly __imp_ names on Windows
// just to work around this issue, let's clone the variable that is constant
// after initialization anyways.

extern "C" {
__declspec(dllimport) int __asan_should_detect_stack_use_after_return();
int __asan_option_detect_stack_use_after_return;

__declspec(dllimport) void *__asan_get_shadow_memory_dynamic_address();
void *__asan_shadow_memory_dynamic_address;

static void __asan_initialize_cloned_variables() {
  __asan_option_detect_stack_use_after_return =
      __asan_should_detect_stack_use_after_return();
  __asan_shadow_memory_dynamic_address =
      __asan_get_shadow_memory_dynamic_address();
}
}

static int asan_thunk_init() {
  __asan_initialize_cloned_variables();

#  ifdef SANITIZER_STATIC_RUNTIME_THUNK
  __asan_initialize_static_thunk();
#  endif

  return 0;
}

static void WINAPI asan_thread_init(void *mod, unsigned long reason,
                                    void *reserved) {
  if (reason == DLL_PROCESS_ATTACH) {
    asan_thunk_init();
  }
}

// Our cloned variables must be initialized before C/C++ constructors.  If TLS
// is used, our .CRT$XLAB initializer will run first. If not, our .CRT$XIB
// initializer is needed as a backup.
extern "C" __declspec(allocate(".CRT$XIB")) int (*__asan_thunk_init)() =
    asan_thunk_init;
WIN_FORCE_LINK(__asan_thunk_init);

extern "C" __declspec(allocate(".CRT$XLAB")) void(WINAPI *__asan_tls_init)(
    void *, unsigned long, void *) = asan_thread_init;
WIN_FORCE_LINK(__asan_tls_init);

////////////////////////////////////////////////////////////////////////////////
// ASan SEH handling.
// We need to set the ASan-specific SEH handler at the end of CRT initialization
// of each module (see also asan_win.cpp).
extern "C" {
__declspec(dllimport) int __asan_set_seh_filter();
static int SetSEHFilter() { return __asan_set_seh_filter(); }

// Unfortunately, putting a pointer to __asan_set_seh_filter into
// __asan_intercept_seh gets optimized out, so we have to use an extra function.
extern "C" __declspec(allocate(".CRT$XCAB")) int (*__asan_seh_interceptor)() =
    SetSEHFilter;
WIN_FORCE_LINK(__asan_seh_interceptor);
}

WIN_FORCE_LINK(__asan_dso_reg_hook)

#endif  // defined(SANITIZER_DYNAMIC_RUNTIME_THUNK) ||
        // defined(SANITIZER_STATIC_RUNTIME_THUNK)
