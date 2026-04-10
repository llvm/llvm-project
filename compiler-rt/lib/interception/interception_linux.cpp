//===-- interception_linux.cpp ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of AddressSanitizer, an address sanity checker.
//
// Linux-specific interception methods.
//
// POSIX dynamic-library helpers (dlopen / dlsym) live here for every
// non-Windows interception target (Linux, *BSD, Darwin, AIX, Fuchsia, ...).
// macOS/AIX/Fuchsia compile this TU for RTInterception but do not use the
// Linux-specific InterceptFunction helpers below.
//===----------------------------------------------------------------------===//

#include "interception.h"

#if !SANITIZER_WINDOWS

#  include <dlfcn.h>

#  pragma weak dlopen
#  pragma weak dlsym
#  pragma weak dlvsym

namespace __interception {

bool DynamicLoaderAvailable() { return dlopen != nullptr && dlsym != nullptr; }

void* OpenLibrary(const char* name) {
  if (!DynamicLoaderAvailable())
    return nullptr;
  return dlopen(name, RTLD_LAZY | RTLD_LOCAL);
}

void* LookupSymbol(void* handle, const char* symbol) {
  if (!DynamicLoaderAvailable())
    return nullptr;
  return dlsym(handle, symbol);
}

void* LookupSymbolDefault(const char* symbol) {
  if (!DynamicLoaderAvailable())
    return nullptr;
  return dlsym(RTLD_DEFAULT, symbol);
}

void* LookupSymbolNext(const char* symbol) {
  if (!DynamicLoaderAvailable())
    return nullptr;
  return dlsym(RTLD_NEXT, symbol);
}

void* LookupSymbolNextVersioned(const char* symbol, const char* version) {
  if (!DynamicLoaderAvailable() || dlvsym == nullptr)
    return nullptr;
  return dlvsym(RTLD_NEXT, symbol, version);
}

}  // namespace __interception

#endif  // !SANITIZER_WINDOWS

#if SANITIZER_LINUX || SANITIZER_FREEBSD || SANITIZER_NETBSD || \
    SANITIZER_SOLARIS || SANITIZER_HAIKU

namespace __interception {

#if SANITIZER_NETBSD
static int StrCmp(const char *s1, const char *s2) {
  while (true) {
    if (*s1 != *s2)
      return false;
    if (*s1 == 0)
      return true;
    s1++;
    s2++;
  }
}
#endif

static void *GetFuncAddr(const char *name, uptr trampoline) {
#if SANITIZER_NETBSD
  // FIXME: Find a better way to handle renames
  if (StrCmp(name, "sigaction"))
    name = "__sigaction14";
#endif
  void* addr = LookupSymbolNext(name);
  if (!addr) {
    // If the lookup using RTLD_NEXT failed, the sanitizer runtime library is
    // later in the library search order than the DSO that we are trying to
    // intercept, which means that we cannot intercept this function. We still
    // want the address of the real definition, though, so look it up using
    // RTLD_DEFAULT.
    addr = LookupSymbolDefault(name);

    // In case `name' is not loaded, dlsym ends up finding the actual wrapper.
    // We don't want to intercept the wrapper and have it point to itself.
    if ((uptr)addr == trampoline)
      addr = nullptr;
  }
  return addr;
}

bool InterceptFunction(const char *name, uptr *ptr_to_real, uptr func,
                       uptr trampoline) {
  void *addr = GetFuncAddr(name, trampoline);
  *ptr_to_real = (uptr)addr;
  return addr && (func == trampoline);
}

// dlvsym is a GNU extension supported by some other platforms.
#if SANITIZER_GLIBC || SANITIZER_FREEBSD || SANITIZER_NETBSD
static void *GetFuncAddr(const char *name, const char *ver) {
  return LookupSymbolNextVersioned(name, ver);
}

bool InterceptFunction(const char *name, const char *ver, uptr *ptr_to_real,
                       uptr func, uptr trampoline) {
  void *addr = GetFuncAddr(name, ver);
  *ptr_to_real = (uptr)addr;
  return addr && (func == trampoline);
}
#  endif  // SANITIZER_GLIBC || SANITIZER_FREEBSD || SANITIZER_NETBSD

}  // namespace __interception

#endif  // SANITIZER_LINUX || SANITIZER_FREEBSD || SANITIZER_NETBSD ||
        // SANITIZER_SOLARIS || SANITIZER_HAIKU
