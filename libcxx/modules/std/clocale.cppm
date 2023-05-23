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
#  include <clocale>
#endif

export module std:clocale;
#ifndef _LIBCPP_HAS_NO_LOCALIZATION
export namespace std {
  using std::lconv;

  using std::localeconv;
  using std::setlocale;

} // namespace std
#endif // _LIBCPP_HAS_NO_LOCALIZATION
