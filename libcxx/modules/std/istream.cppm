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
#  include <istream>
#endif

export module std:istream;
#ifndef _LIBCPP_HAS_NO_LOCALIZATION
export namespace std {
  using std::basic_istream;

  using std::istream;
#  ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
  using std::wistream;
#  endif

  using std::basic_iostream;

  using std::iostream;
#  ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
  using std::wiostream;
#  endif

  using std::ws;

  using std::operator>>;
} // namespace std
#endif // _LIBCPP_HAS_NO_LOCALIZATION
