//===-- Implementation header for fegetexceptflag ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_FENV_FEGETEXCEPTFLAG_H
#define LLVM_LIBC_SRC_FENV_FEGETEXCEPTFLAG_H

#include <fenv.h>

namespace __llvm_libc {

int fegetexceptflag(fexcept_t *, int excepts);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_FENV_FEGETEXCEPTFLAG_H
