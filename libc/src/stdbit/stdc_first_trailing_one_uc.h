//===-- Implementation header for stdc_first_trailing_one_uc ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDBIT_STDC_FIRST_TRAILING_ONE_UC_H
#define LLVM_LIBC_SRC_STDBIT_STDC_FIRST_TRAILING_ONE_UC_H

#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

unsigned stdc_first_trailing_one_uc(unsigned char value);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STDBIT_STDC_FIRST_TRAILING_ONE_UC_H
