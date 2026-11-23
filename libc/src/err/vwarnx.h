//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation header for vwarnx.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_ERR_VWARNX_H
#define LLVM_LIBC_SRC_ERR_VWARNX_H

#include "src/__support/macros/config.h"
#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

void vwarnx(const char *fmt, va_list args);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_ERR_VWARNX_H
