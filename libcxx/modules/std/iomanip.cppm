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
#  include <iomanip>
#endif

export module std:iomanip;
#ifndef _LIBCPP_HAS_NO_LOCALIZATION
export namespace std {
  using std::get_money;
  using std::get_time;
  using std::put_money;
  using std::put_time;
  using std::resetiosflags;
  using std::setbase;
  using std::setfill;
  using std::setiosflags;
  using std::setprecision;
  using std::setw;

  using std::quoted;
} // namespace std
#endif // _LIBCPP_HAS_NO_LOCALIZATION
