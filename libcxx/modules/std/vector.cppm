// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

module;
#include <functional>
#include <vector>

export module std:vector;
export namespace std {
  // [vector], class template vector
  using std::vector;

  using std::operator==;
#if 0 // P1614
  using std::operator<=>;
#else
  /* These should be removed after https://reviews.llvm.org/D132268 lands. */
  using std::operator!=;
  using std::operator<;
  using std::operator>;
  using std::operator<=;
  using std::operator>=;
#endif

  using std::swap;

  // [vector.erasure], erasure
  using std::erase;
  using std::erase_if;

  namespace pmr {
    using std::pmr::vector;
  }

  // hash support
  using std::hash;

  // [vector.bool.fmt], formatter specialization for vector<bool>
  using std::formatter;
} // namespace std
