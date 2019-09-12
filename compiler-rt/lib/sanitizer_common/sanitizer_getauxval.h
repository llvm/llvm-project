//===-- sanitizer_getauxval.h -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common getauxval() guards and definitions.
// getauxval() is not defined until glibc version 2.16, or until API level 21
// for Android.
//
//===----------------------------------------------------------------------===//

#ifndef SANITIZER_GETAUXVAL_H
#define SANITIZER_GETAUXVAL_H

#include "sanitizer_platform.h"

#if SANITIZER_LINUX || SANITIZER_FUCHSIA

# include <features.h>

# ifndef __GLIBC_PREREQ
#  define __GLIBC_PREREQ(x, y) 0
# endif

# if __GLIBC_PREREQ(2, 16) || (SANITIZER_ANDROID && __ANDROID_API__ >= 21) || \
     SANITIZER_FUCHSIA
#  define SANITIZER_USE_GETAUXVAL 1
# else
#  define SANITIZER_USE_GETAUXVAL 0
# endif

# if SANITIZER_USE_GETAUXVAL
#  include <sys/auxv.h>
# else
// The weak getauxval definition allows to check for the function at runtime.
// This is useful for Android, when compiled at a lower API level yet running
// on a more recent platform that offers the function.
extern "C" SANITIZER_WEAK_ATTRIBUTE unsigned long getauxval(unsigned long type);
# endif

#endif // SANITIZER_LINUX || SANITIZER_FUCHSIA

#endif // SANITIZER_GETAUXVAL_H
