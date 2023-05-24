// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module;
#if __has_include(<spanstream>)
#  error "include this header unconditionally and uncomment the exported symbols"
#  include <spanstream>
#endif

export module std:spanstream;
export namespace std {
#if 0
  using std::basic_spanbuf;

  using std::swap;

  using std::spanbuf;
#  ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
  using std::wspanbuf;
#  endif

  using std::basic_ispanstream;

  using std::ispanstream;
#  ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
  using std::wispanstream;
#  endif

  using std::basic_ospanstream;

  using std::ospanstream;
#  ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
  using std::wospanstream;
#  endif

  using std::basic_spanstream;

  using std::spanstream;
#  ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
  using std::wspanstream;
#  endif
#endif
} // namespace std
