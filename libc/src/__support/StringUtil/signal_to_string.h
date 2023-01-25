//===-- Function prototype for mapping signals to strings -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"

#ifndef LLVM_LIBC_SRC_SUPPORT_SIGNAL_TO_STRING
#define LLVM_LIBC_SRC_SUPPORT_SIGNAL_TO_STRING

namespace __llvm_libc {

cpp::string_view get_signal_string(int err_num);

cpp::string_view get_signal_string(int err_num, cpp::span<char> buffer);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_SIGNAL_TO_STRING
