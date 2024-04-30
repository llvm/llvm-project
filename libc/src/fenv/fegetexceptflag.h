//===-- Implementation header for fegetexceptflag ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_FENV_FEGETEXCEPTFLAG_H
#define LLVM_LIBC_SRC_FENV_FEGETEXCEPTFLAG_H

#include "hdr/types/fexcept_t.h"

namespace LIBC_NAMESPACE {

int fegetexceptflag(fexcept_t *, int excepts);

} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_FENV_FEGETEXCEPTFLAG_H
