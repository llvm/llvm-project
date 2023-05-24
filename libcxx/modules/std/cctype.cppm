// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module;
#include <cctype>

export module std:cctype;
export namespace std {
  using std::isalnum;
  using std::isalpha;
  using std::isblank;
  using std::iscntrl;
  using std::isdigit;
  using std::isgraph;
  using std::islower;
  using std::isprint;
  using std::ispunct;
  using std::isspace;
  using std::isupper;
  using std::isxdigit;
  using std::tolower;
  using std::toupper;
} // namespace std
