//===--------------- Virtual DSO Support ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_VDSO_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_VDSO_H

#if defined(__linux__)
#include "linux/vdso.h"
#elif defined(__Fuchsia__)
#include "fuchsia/vdso.h"
#endif

#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_VDSO_H
