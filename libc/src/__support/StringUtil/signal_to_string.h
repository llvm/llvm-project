//===-- Function prototype for mapping signals to strings -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


#ifndef LLVM_LIBC_SRC___SUPPORT_STRINGUTIL_SIGNAL_TO_STRING_H
#define LLVM_LIBC_SRC___SUPPORT_STRINGUTIL_SIGNAL_TO_STRING_H

#include "src/__support/CPP/span.h"
#include "src/__support/CPP/string_view.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// The description of a signal. The view is not followed by a terminator in
// general, so a caller that needs a C string uses get_signal_cstring. If the
// signal has no description of its own the text is built in the buffer, or in
// a thread local one if none is given, and is valid until the next call.
cpp::string_view get_signal_string(int sig_num);

cpp::string_view get_signal_string(int sig_num, cpp::span<char> buffer);

// As get_signal_string, but the result is a terminated C string.
const char *get_signal_cstring(int sig_num);

const char *get_signal_cstring(int sig_num, cpp::span<char> buffer);

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_STRINGUTIL_SIGNAL_TO_STRING_H
