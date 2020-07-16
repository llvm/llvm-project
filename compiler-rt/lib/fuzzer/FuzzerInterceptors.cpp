//===-- FuzzerInterceptors.cpp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Intercept certain libc functions to aid fuzzing.
// Linked only when other RTs that define their own interceptors are not linked.
//===----------------------------------------------------------------------===//

#include "FuzzerPlatform.h"

#if LIBFUZZER_LINUX

#define GET_CALLER_PC() __builtin_return_address(0)

#define PTR_TO_REAL(x) real_##x
#define REAL(x) __interception::PTR_TO_REAL(x)
#define FUNC_TYPE(x) x##_type
#define DEFINE_REAL(ret_type, func, ...)                                       \
  typedef ret_type (*FUNC_TYPE(func))(__VA_ARGS__);                            \
  namespace __interception {                                                   \
  FUNC_TYPE(func) PTR_TO_REAL(func);                                           \
  }

#include <cassert>
#include <cstdint>
#include <dlfcn.h> // for dlsym()
#include <sanitizer/common_interface_defs.h>

static void *getFuncAddr(const char *name, uintptr_t wrapper_addr) {
  void *addr = dlsym(RTLD_NEXT, name);
  if (!addr) {
    // If the lookup using RTLD_NEXT failed, the sanitizer runtime library is
    // later in the library search order than the DSO that we are trying to
    // intercept, which means that we cannot intercept this function. We still
    // want the address of the real definition, though, so look it up using
    // RTLD_DEFAULT.
    addr = dlsym(RTLD_DEFAULT, name);

    // In case `name' is not loaded, dlsym ends up finding the actual wrapper.
    // We don't want to intercept the wrapper and have it point to itself.
    if (reinterpret_cast<uintptr_t>(addr) == wrapper_addr)
      addr = nullptr;
  }
  return addr;
}

static int FuzzerInited = 0;
static bool FuzzerInitIsRunning;

static void fuzzerInit();

static void ensureFuzzerInited() {
  assert(!FuzzerInitIsRunning);
  if (!FuzzerInited) {
    fuzzerInit();
  }
}

extern "C" {

DEFINE_REAL(int, memcmp, const void *, const void *, size_t)
DEFINE_REAL(int, strncmp, const char *, const char *, size_t)
DEFINE_REAL(int, strcmp, const char *, const char *)
DEFINE_REAL(int, strncasecmp, const char *, const char *, size_t)
DEFINE_REAL(int, strcasecmp, const char *, const char *)
DEFINE_REAL(char *, strstr, const char *, const char *)
DEFINE_REAL(char *, strcasestr, const char *, const char *)
DEFINE_REAL(void *, memmem, const void *, size_t, const void *, size_t)

ATTRIBUTE_INTERFACE int memcmp(const void *s1, const void *s2, size_t n) {
  ensureFuzzerInited();
  int result = REAL(memcmp)(s1, s2, n);
  __sanitizer_weak_hook_memcmp(GET_CALLER_PC(), s1, s2, n, result);

  return result;
}

ATTRIBUTE_INTERFACE int strncmp(const char *s1, const char *s2, size_t n) {
  ensureFuzzerInited();
  int result = REAL(strncmp)(s1, s2, n);
  __sanitizer_weak_hook_strncmp(GET_CALLER_PC(), s1, s2, n, result);

  return result;
}

ATTRIBUTE_INTERFACE int strcmp(const char *s1, const char *s2) {
  ensureFuzzerInited();
  int result = REAL(strcmp)(s1, s2);
  __sanitizer_weak_hook_strcmp(GET_CALLER_PC(), s1, s2, result);

  return result;
}

ATTRIBUTE_INTERFACE int strncasecmp(const char *s1, const char *s2, size_t n) {
  ensureFuzzerInited();
  int result = REAL(strncasecmp)(s1, s2, n);
  __sanitizer_weak_hook_strncasecmp(GET_CALLER_PC(), s1, s2, n, result);

  return result;
}

ATTRIBUTE_INTERFACE int strcasecmp(const char *s1, const char *s2) {
  ensureFuzzerInited();
  int result = REAL(strcasecmp)(s1, s2);
  __sanitizer_weak_hook_strcasecmp(GET_CALLER_PC(), s1, s2, result);

  return result;
}

ATTRIBUTE_INTERFACE char *strstr(const char *s1, const char *s2) {
  ensureFuzzerInited();
  char *result = REAL(strstr)(s1, s2);
  __sanitizer_weak_hook_strstr(GET_CALLER_PC(), s1, s2, result);

  return result;
}

ATTRIBUTE_INTERFACE char *strcasestr(const char *s1, const char *s2) {
  ensureFuzzerInited();
  char *result = REAL(strcasestr)(s1, s2);
  __sanitizer_weak_hook_strcasestr(GET_CALLER_PC(), s1, s2, result);

  return result;
}

ATTRIBUTE_INTERFACE
void *memmem(const void *s1, size_t len1, const void *s2, size_t len2) {
  ensureFuzzerInited();
  void *result = REAL(memmem)(s1, len1, s2, len2);
  __sanitizer_weak_hook_memmem(GET_CALLER_PC(), s1, len1, s2, len2, result);

  return result;
}

__attribute__((section(".preinit_array"),
               used)) static void (*__local_fuzzer_preinit)(void) = fuzzerInit;

} // extern "C"

static void fuzzerInit() {
  assert(!FuzzerInitIsRunning);
  if (FuzzerInited)
    return;
  FuzzerInitIsRunning = true;

  REAL(memcmp) = reinterpret_cast<memcmp_type>(
      getFuncAddr("memcmp", reinterpret_cast<uintptr_t>(&memcmp)));
  REAL(strncmp) = reinterpret_cast<strncmp_type>(
      getFuncAddr("strncmp", reinterpret_cast<uintptr_t>(&strncmp)));
  REAL(strcmp) = reinterpret_cast<strcmp_type>(
      getFuncAddr("strcmp", reinterpret_cast<uintptr_t>(&strcmp)));
  REAL(strncasecmp) = reinterpret_cast<strncasecmp_type>(
      getFuncAddr("strncasecmp", reinterpret_cast<uintptr_t>(&strncasecmp)));
  REAL(strcasecmp) = reinterpret_cast<strcasecmp_type>(
      getFuncAddr("strcasecmp", reinterpret_cast<uintptr_t>(&strcasecmp)));
  REAL(strstr) = reinterpret_cast<strstr_type>(
      getFuncAddr("strstr", reinterpret_cast<uintptr_t>(&strstr)));
  REAL(strcasestr) = reinterpret_cast<strcasestr_type>(
      getFuncAddr("strcasestr", reinterpret_cast<uintptr_t>(&strcasestr)));
  REAL(memmem) = reinterpret_cast<memmem_type>(
      getFuncAddr("memmem", reinterpret_cast<uintptr_t>(&memmem)));

  FuzzerInitIsRunning = false;
  FuzzerInited = 1;
}

#endif
