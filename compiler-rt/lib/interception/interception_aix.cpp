//===-- interception_aix.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// AIX-specific interception methods.
//===----------------------------------------------------------------------===//

#include "interception.h"
#include "sanitizer_common/sanitizer_common.h"

#if SANITIZER_AIX

#  include <dlfcn.h>  // for dlsym()
#  include <stddef.h> // for size_t

#if SANITIZER_WORDSIZE == 64
#define STRCPY_STR "___strcpy64"
#define MEMCPY_STR "___memcpy64"
#define MEMMOVE_STR "___memmove64"
#else
#define STRCPY_STR "___strcpy"
#define MEMCPY_STR "___memcpy"
#define MEMMOVE_STR "___memmove"
#endif

namespace __interception {

char *___strcpy(char *, const char *) __asm__(STRCPY_STR);
char *___memcpy(char *, const char *, size_t) __asm__(MEMCPY_STR);
char *___memmove(char *, const char *, size_t) __asm__(MEMMOVE_STR);

char* real_strcpy_wrapper(char *s1, const char *s2) {
  return (char*)___strcpy(s1, s2);
}

char* real_memcpy_wrapper(char *s1, const char *s2, size_t n) {
  return (char*)___memcpy(s1, s2, n);
}

char* real_memmove_wrapper(char *s1, const char *s2, size_t n) {
  return (char*)___memmove(s1, s2, n);
}


static void *GetFuncAddr(const char *name, uptr wrapper_addr) {
  // FIXME: if we are going to ship dynamic asan library, we may need to search
  // all the loaded modules with RTLD_DEFAULT if RTLD_NEXT failed.
  void *addr = dlsym(RTLD_NEXT, name);

  // AIX dlsym can only detect functions that are exported, so
  // some basic functions like memcpy return null. In this case, we fallback
  // to a corresponding internal libc symbol (for example, ___memcpy) if it's 
  // available, or the internal sanitizer function.
  if (!addr) {
    if (internal_strcmp(name, "strcpy") == 0)
      addr = (void*)real_strcpy_wrapper;
    else if (internal_strcmp(name, "strncpy") == 0)
      addr = (void*)internal_strncpy;
    else if (internal_strcmp(name, "strcat") == 0)
      addr = (void*)internal_strcat;
    else if (internal_strcmp(name, "strncat") == 0)
      addr = (void*)internal_strncat;
    else if (internal_strcmp(name, "memcpy") == 0)
      addr = (void*)real_memcpy_wrapper;
    else if (internal_strcmp(name, "memmove") == 0)
      addr = (void*)real_memmove_wrapper;
  }

  // In case `name' is not loaded, dlsym ends up finding the actual wrapper.
  // We don't want to intercept the wrapper and have it point to itself.
  if ((uptr)addr == wrapper_addr)
    addr = nullptr;
  return addr;
}

bool InterceptFunction(const char *name, uptr *ptr_to_real, uptr func,
                       uptr wrapper) {
  void *addr = GetFuncAddr(name, wrapper);
  *ptr_to_real = (uptr)addr;
  return addr && (func == wrapper);
}

}  // namespace __interception
#endif  // SANITIZER_AIX
