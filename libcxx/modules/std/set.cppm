// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module;
#include <set>

export module std:set;
export namespace std {
  // [set], class template set
  using std::set;

  using std::operator==;
#if 0 // P1614
  using std::operator<=>;
#else
  using std::operator!=;
  using std::operator<;
  using std::operator>;
  using std::operator<=;
  using std::operator>=;
#endif

  using std::swap;

  // [set.erasure], erasure for set
  using std::erase_if;

  // [multiset], class template multiset
  using std::multiset;

  namespace pmr {
    using std::pmr::multiset;
    using std::pmr::set;
  } // namespace pmr
} // namespace std
