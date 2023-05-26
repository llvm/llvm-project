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
#  include <codecvt>
#endif

export module std:codecvt;
#ifndef _LIBCPP_HAS_NO_LOCALIZATION
export namespace std {

  using std::codecvt_mode;

  using std::codecvt_utf16;
  using std::codecvt_utf8;
  using std::codecvt_utf8_utf16;

} // namespace std
#endif // _LIBCPP_HAS_NO_LOCALIZATION
