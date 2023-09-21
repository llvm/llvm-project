//===-- Definition of a class for mapping errors to strings -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_STRINGUTIL_ERROR_TO_STRING_H
#define LLVM_LIBC_SRC___SUPPORT_STRINGUTIL_ERROR_TO_STRING_H

#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"

namespace __llvm_libc {

cpp::string_view get_error_string(int err_num);

cpp::string_view get_error_string(int err_num, cpp::span<char> buffer);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC___SUPPORT_STRINGUTIL_ERROR_TO_STRING_H
