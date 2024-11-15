//===-- Including wchar.h in overlay mode ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_HDR_WCHAR_OVERLAY_H
#define LLVM_LIBC_HDR_WCHAR_OVERLAY_H

#ifdef LIBC_FULL_BUILD
#error "This header should only be included in overlay mode"
#endif

// Overlay mode

// glibc <wchar.h header might provide extern inline definitions for few
// functions, causing external alias errors.  They are guarded by
// `__USE_EXTERN_INLINES` macro.  We temporarily disable `__USE_EXTERN_INLINES`
// macro by defining `__NO_INLINE__` before including <wchar.h>.
// And the same with `__USE_FORTIFY_LEVEL`, which will be temporarily disabled
// with `_FORTIFY_SOURCE`.

#ifdef _FORTIFY_SOURCE
#define LIBC_OLD_FORTIFY_SOURCE _FORTIFY_SOURCE
#undef _FORTIFY_SOURCE
#endif

#ifndef __NO_INLINE__
#define __NO_INLINE__ 1
#define LIBC_SET_NO_INLINE
#endif

#include <wchar.h>

#ifdef LIBC_OLD_FORTIFY_SOURCE
#define _FORTIFY_SOURCE LIBC_OLD_FORTIFY_SOURCE
#undef LIBC_OLD_FORTIFY_SOURCE
#endif

#ifdef LIBC_SET_NO_INLINE
#undef __NO_INLINE__
#undef LIBC_SET_NO_INLINE
#endif

#endif // LLVM_LIBC_HDR_WCHAR_OVERLAY_H
