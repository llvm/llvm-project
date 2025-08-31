//===-- Implementation header of localtime_r --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_LOCALTIME_R_H
#define LLVM_LIBC_SRC_TIME_LOCALTIME_R_H

#include "src/__support/macros/config.h"
#include <time.h>

namespace LIBC_NAMESPACE_DECL {

struct tm *localtime_r(const time_t *t_ptr, struct tm *input);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_TIME_LOCALTIME_R_H
