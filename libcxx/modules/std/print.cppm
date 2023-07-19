// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module;
#include <print>

export module std:print;
export namespace std {
  // [print.fun], print functions
  using std::print;
  using std::println;

  using std::vprint_nonunicode;
  using std::vprint_unicode;
} // namespace std
