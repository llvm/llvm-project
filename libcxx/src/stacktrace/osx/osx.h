//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP_STACKTRACE_OSX_H
#define _LIBCPP_STACKTRACE_OSX_H

#include <__config>
#include <__config_site>
#include <cstddef>
#include <cstdlib>

_LIBCPP_BEGIN_NAMESPACE_STD
namespace __stacktrace {

struct context;

struct osx {
  context& cx_;
  void ident_modules();
  void symbolize();
};

} // namespace __stacktrace
_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STACKTRACE_OSX_H
