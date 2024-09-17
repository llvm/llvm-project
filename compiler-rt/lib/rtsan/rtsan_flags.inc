//===------------------------ rtsan_flags.inc -------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// RTSan runtime flags.
//
//===----------------------------------------------------------------------===//
#ifndef RTSAN_FLAG
#error "Define RTSAN_FLAG prior to including this file!"
#endif

// RTSAN_FLAG(Type, Name, DefaultValue, Description)
// See COMMON_FLAG in sanitizer_flags.inc for more details.

// Example flag, until we get a real one
// RTSAN_FLAG(bool, halt_on_error, true, "If true, halt the program on error")
