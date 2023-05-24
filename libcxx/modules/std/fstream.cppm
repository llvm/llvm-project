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
#  include <fstream>
#endif

export module std:fstream;
#ifndef _LIBCPP_HAS_NO_LOCALIZATION
export namespace std {
  using std::basic_filebuf;

  using std::swap;

  using std::filebuf;
#  ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
  using std::wfilebuf;
#  endif

  using std::basic_ifstream;

  using std::ifstream;
#  ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
  using std::wifstream;
#  endif

  using std::basic_ofstream;

  using std::ofstream;
#  ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
  using std::wofstream;
#  endif

  using std::basic_fstream;

  using std::fstream;
#  ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
  using std::wfstream;
#  endif
} // namespace std
#endif // _LIBCPP_HAS_NO_LOCALIZATION
