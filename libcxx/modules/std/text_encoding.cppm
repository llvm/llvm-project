// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module;
#if __has_include(<text_encoding>)
#  error "include this header unconditionally and uncomment the exported symbols"
#  include <text_encoding>
#endif

export module std:text_encoding;
export namespace std {
#if 0
#  if _LIBCPP_STD_VER >= 23
  using std::text_encoding;

  // hash support
  using std::hash;
#  endif // _LIBCPP_STD_VER >= 23
#endif
} // namespace std
