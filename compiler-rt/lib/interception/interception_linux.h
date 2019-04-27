//===-- interception_linux.h ------------------------------------*- C++ -*-===//
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
//===----------------------------------------------------------------------===//

#if SANITIZER_LINUX || SANITIZER_FREEBSD || SANITIZER_NETBSD || \
    SANITIZER_OPENBSD || SANITIZER_SOLARIS

#if !defined(INCLUDED_FROM_INTERCEPTION_LIB)
# error "interception_linux.h should be included from interception library only"
#endif

#ifndef INTERCEPTION_LINUX_H
#define INTERCEPTION_LINUX_H

namespace __interception {
void *GetFuncAddr(const char *name);
void *GetFuncAddrVer(const char *name, const char *ver);
}  // namespace __interception

#define INTERCEPT_FUNCTION_LINUX_OR_FREEBSD(func)                          \
  (REAL(func) = (FUNC_TYPE(func)) ::__interception::GetFuncAddr(#func))

// Android,  Solaris and OpenBSD do not have dlvsym
#if !SANITIZER_ANDROID && !SANITIZER_SOLARIS && !SANITIZER_OPENBSD
#define INTERCEPT_FUNCTION_VER_LINUX_OR_FREEBSD(func, symver)              \
  (REAL(func) =                                                            \
      (FUNC_TYPE(func)) ::__interception::GetFuncAddrVer(#func, symver))
#else
#define INTERCEPT_FUNCTION_VER_LINUX_OR_FREEBSD(func, symver) \
  INTERCEPT_FUNCTION_LINUX_OR_FREEBSD(func)
#endif  // !SANITIZER_ANDROID && !SANITIZER_SOLARIS

#endif  // INTERCEPTION_LINUX_H
#endif  // SANITIZER_LINUX || SANITIZER_FREEBSD || SANITIZER_NETBSD ||
        // SANITIZER_OPENBSD || SANITIZER_SOLARIS
