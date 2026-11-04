//===--- system-other.h - Stub header for tests -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _SYSTEM_OTHER_H_
#define _SYSTEM_OTHER_H_

// Special system calls.

#if __STDC_VERSION__ < 202311L
void other_call();
#endif

#endif // _SYSTEM_OTHER_H_
