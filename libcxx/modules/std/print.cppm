// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module;
#if __has_include(<print>)
#  error "include this header unconditionally and uncomment the exported symbols"
#  include <print>
#endif

export module std:print;
export namespace std {
#if 0
  // [print.fun], print functions
  using std::print;
  using std::println;

  using std::vprint_nonunicode;
  using std::vprint_unicode;
#endif
} // namespace std
