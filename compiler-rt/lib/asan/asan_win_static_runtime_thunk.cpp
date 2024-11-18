//===-- asan_win_static_runtime_thunk.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// This file defines a family of thunks that should be statically linked into
// modules that are statically linked with the C Runtime in order to delegate
// the calls to the ASAN runtime DLL.
// See https://github.com/google/sanitizers/issues/209 for the details.
//===----------------------------------------------------------------------===//

#ifdef SANITIZER_STATIC_RUNTIME_THUNK
#  include "asan_init_version.h"
#  include "asan_interface_internal.h"
#  include "asan_win_common_runtime_thunk.h"
#  include "sanitizer_common/sanitizer_platform_interceptors.h"
#  include "sanitizer_common/sanitizer_win_defs.h"
#  include "sanitizer_common/sanitizer_win_thunk_interception.h"

#  if defined(_MSC_VER) && !defined(__clang__)
// Disable warnings such as: 'void memchr(void)': incorrect number of arguments
// for intrinsic function, expected '3' arguments.
#    pragma warning(push)
#    pragma warning(disable : 4392)
#  endif

#  define INTERCEPT_LIBRARY_FUNCTION_ASAN(X) \
    INTERCEPT_LIBRARY_FUNCTION(X, "__asan_wrap_" #X)

INTERCEPT_LIBRARY_FUNCTION_ASAN(atoi);
INTERCEPT_LIBRARY_FUNCTION_ASAN(atol);
INTERCEPT_LIBRARY_FUNCTION_ASAN(atoll);
INTERCEPT_LIBRARY_FUNCTION_ASAN(frexp);
INTERCEPT_LIBRARY_FUNCTION_ASAN(longjmp);
#  if SANITIZER_INTERCEPT_MEMCHR
INTERCEPT_LIBRARY_FUNCTION_ASAN(memchr);
#  endif
INTERCEPT_LIBRARY_FUNCTION_ASAN(memcmp);
INTERCEPT_LIBRARY_FUNCTION_ASAN(memcpy);
#  ifndef _WIN64
// memmove and memcpy share an implementation on amd64
INTERCEPT_LIBRARY_FUNCTION_ASAN(memmove);
#  endif
INTERCEPT_LIBRARY_FUNCTION_ASAN(memset);
INTERCEPT_LIBRARY_FUNCTION_ASAN(strcat);
INTERCEPT_LIBRARY_FUNCTION_ASAN(strchr);
INTERCEPT_LIBRARY_FUNCTION_ASAN(strcmp);
INTERCEPT_LIBRARY_FUNCTION_ASAN(strcpy);
INTERCEPT_LIBRARY_FUNCTION_ASAN(strcspn);
INTERCEPT_LIBRARY_FUNCTION_ASAN(_strdup);
INTERCEPT_LIBRARY_FUNCTION_ASAN(strlen);
INTERCEPT_LIBRARY_FUNCTION_ASAN(strncat);
INTERCEPT_LIBRARY_FUNCTION_ASAN(strncmp);
INTERCEPT_LIBRARY_FUNCTION_ASAN(strncpy);
INTERCEPT_LIBRARY_FUNCTION_ASAN(strnlen);
INTERCEPT_LIBRARY_FUNCTION_ASAN(strpbrk);
// INTERCEPT_LIBRARY_FUNCTION_ASAN(strrchr);
INTERCEPT_LIBRARY_FUNCTION_ASAN(strspn);
INTERCEPT_LIBRARY_FUNCTION_ASAN(strstr);
INTERCEPT_LIBRARY_FUNCTION_ASAN(strtok);
INTERCEPT_LIBRARY_FUNCTION_ASAN(wcslen);
INTERCEPT_LIBRARY_FUNCTION_ASAN(wcsnlen);
INTERCEPT_LIBRARY_FUNCTION_ASAN(_aligned_malloc);
INTERCEPT_LIBRARY_FUNCTION_ASAN(_aligned_realloc);
INTERCEPT_LIBRARY_FUNCTION_ASAN(_aligned_free);
INTERCEPT_LIBRARY_FUNCTION_ASAN(_aligned_msize);

// Note: Don't intercept strtol(l). They are supposed to set errno for out-of-
// range values, but since the ASan runtime is linked against the dynamic CRT,
// its errno is different from the one in the current module.

#  if defined(_MSC_VER) && !defined(__clang__)
#    pragma warning(pop)
#  endif

#  ifdef _WIN64
INTERCEPT_LIBRARY_FUNCTION_ASAN(__C_specific_handler);
#  else
extern "C" void abort();
INTERCEPT_LIBRARY_FUNCTION_ASAN(_except_handler3);
// _except_handler4 checks -GS cookie which is different for each module, so we
// can't use INTERCEPT_LIBRARY_FUNCTION_ASAN(_except_handler4), need to apply
// manually
extern "C" int _except_handler4(void *, void *, void *, void *);
static int (*real_except_handler4)(void *, void *, void *,
                                   void *) = &_except_handler4;
static int intercept_except_handler4(void *a, void *b, void *c, void *d) {
  __asan_handle_no_return();
  return real_except_handler4(a, b, c, d);
}
#  endif

// Windows specific functions not included in asan_interface.inc.
// INTERCEPT_WRAP_W_V(__asan_should_detect_stack_use_after_return)
// INTERCEPT_WRAP_W_V(__asan_get_shadow_memory_dynamic_address)
// INTERCEPT_WRAP_W_W(__asan_unhandled_exception_filter)

extern "C" void __asan_initialize_static_thunk() {
#  ifndef _WIN64
  if (real_except_handler4 == &_except_handler4) {
    // Single threaded, no need for synchronization.
    if (!__sanitizer_override_function_by_addr(
            reinterpret_cast<__sanitizer::uptr>(&intercept_except_handler4),
            reinterpret_cast<__sanitizer::uptr>(&_except_handler4),
            reinterpret_cast<__sanitizer::uptr*>(&real_except_handler4))) {
      abort();
    }
  }
#  endif
}

#endif  // SANITIZER_DLL_THUNK
