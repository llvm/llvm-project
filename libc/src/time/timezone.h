//===-- Implementation of timezone functions ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_TIME_TIMEZONE_H
#define LLVM_LIBC_SRC_TIME_TIMEZONE_H

#include <stddef.h> // For size_t.

#include "src/__support/CPP/limits.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/errno/libc_errno.h"
#include "src/time/mktime.h"

#include <stdint.h>

namespace LIBC_NAMESPACE_DECL {
namespace timezone {

extern int get_timezone_offset(char *timezone);

} // namespace timezone
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_TIME_TIMEZONE_H
