//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the prototype for the vswprintf function.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_WCHAR_VSWPRINTF_H
#define LLVM_LIBC_SRC_WCHAR_VSWPRINTF_H

#include "hdr/types/size_t.h"
#include "hdr/types/wchar_t.h"
#include "src/__support/macros/config.h"

#include <stdarg.h>

namespace LIBC_NAMESPACE_DECL {

int vswprintf(wchar_t *__restrict buffer, size_t bufsz,
              const wchar_t *__restrict format, va_list vlist);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_WCHAR_VSWPRINTF_H
