// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module;
#if __has_include(<syncstream>)
#  error "include this header unconditionally and uncomment the exported symbols"
#  include <syncstream>
#endif

export module std:syncstream;
export namespace std {
#if 0
  using std::basic_syncbuf;

  // [syncstream.syncbuf.special], specialized algorithms
  using std::swap;

  using std::syncbuf;
#  ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
  using std::wsyncbuf;
#  endif
  using std::basic_osyncstream;

  using std::osyncstream;
#  ifndef _LIBCPP_HAS_NO_WIDE_CHARACTERS
  using std::wosyncstream;
#  endif
#endif
} // namespace std
