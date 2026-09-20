// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_SYMBOLS_H
#define _LIBCPP_STACKTRACE_SYMBOLS_H

#include <__config>
#include <__stacktrace/basic_stacktrace.h>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

void __populate_symbols(_Context& __cx);

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_SYMBOLS_H
