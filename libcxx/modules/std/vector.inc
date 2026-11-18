// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

export namespace std {
  // [vector], class template vector
  using std::vector;

  using std::operator==;
  using std::operator<=>;

  using std::swap;

  // [vector.erasure], erasure
  using std::erase;
  using std::erase_if;

  namespace pmr {
    using std::pmr::vector;
  }

  // hash support
  using std::hash;

#if _LIBCPP_STD_VER >= 23
  // [vector.bool.fmt], formatter specialization for vector<bool>
  using std::formatter;
#endif

#if _LIBCPP_STD_VER >= 29
  // PROTOTYPE: std::vector2 from P3866R0
  using std::vector2;

  namespace pmr {
    using std::pmr::vector2;
  }

  // PROTOTYPE: P3866R0 standard-release namespace entries contributed by <vector>
  namespace cpp29 {
    using std::cpp29::vector;

    namespace pmr {
      using std::cpp29::pmr::vector;
    }
  } // namespace cpp29
#endif
} // namespace std
