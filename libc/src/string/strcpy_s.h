//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the implementation header for strcpy_s.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_STRCPY_S_H
#define LLVM_LIBC_SRC_STRING_STRCPY_S_H

#include "hdr/types/errno_t.h"
#include "hdr/types/rsize_t.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

errno_t strcpy_s(char *__restrict s1, rsize_t s1max, const char *__restrict s2);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STRING_STRCPY_S_H
