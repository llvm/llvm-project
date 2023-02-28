// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module;
#include <__config>
#ifndef _LIBCPP_HAS_NO_LOCALIZATION
#  include <streambuf>
#endif

export module std:streambuf;
#ifndef _LIBCPP_HAS_NO_LOCALIZATION
export namespace std {
  using std::basic_streambuf;
  using std::streambuf;
#  ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
  using std::wstreambuf;
#  endif
} // namespace std
#endif // _LIBCPP_HAS_NO_LOCALIZATION
