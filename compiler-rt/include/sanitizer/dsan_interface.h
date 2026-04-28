//===-- sanitizer/dsan_interface.h ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DoubleFreeSanitizer (DSan).
//
// Public interface header.
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_DSAN_INTERFACE_H
#define SANITIZER_DSAN_INTERFACE_H

#include <sanitizer/common_interface_defs.h>

#ifdef __cplusplus
extern "C" {
#endif

// This function may be optionally provided by user and should return
// a string containing common sanitizer runtime options.
const char *SANITIZER_CDECL __dsan_default_options(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // SANITIZER_DSAN_INTERFACE_H
