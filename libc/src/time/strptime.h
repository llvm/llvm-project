//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the declaration of the strptime function.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_STRPTIME_H
#define LLVM_LIBC_SRC_TIME_STRPTIME_H

#include "hdr/types/struct_tm.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

char *strptime(const char *__restrict buf, const char *__restrict format,
               const struct tm *__restrict tm);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_TIME_STRPTIME_H
