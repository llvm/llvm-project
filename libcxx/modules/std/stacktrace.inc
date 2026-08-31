// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

export namespace std {
#if _LIBCPP_STD_VER >= 23 && _LIBCPP_AVAILABILITY_HAS_STACKTRACE

  // [stacktrace.entry], class stacktrace_entry
  using std::stacktrace_entry;

  // [stacktrace.basic], class template basic_stacktrace
  using std::basic_stacktrace;

  // basic_stacktrace typedef-names
  using std::stacktrace;

  // [stacktrace.basic.nonmem], non-member functions
  using std::swap;

#  if _LIBCPP_HAS_LOCALIZATION
  using std::to_string;
  using std::operator<<;
#  endif // _LIBCPP_HAS_LOCALIZATION

  namespace pmr {
    using std::pmr::stacktrace;
  }

  // [stacktrace.basic.hash], hash support
  using std::hash;

  // [stacktrace.format], formatting support
  // (Excluded for now; issue #105257 will implement
  // P2693R1, "Formatting thread::id and stacktrace")
  // using std::formatter;

#endif // _LIBCPP_STD_VER >= 23 && _LIBCPP_AVAILABILITY_HAS_STACKTRACE
} // namespace std
