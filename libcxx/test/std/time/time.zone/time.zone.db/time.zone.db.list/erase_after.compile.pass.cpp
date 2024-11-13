//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-experimental-tzdb
// XFAIL: availability-tzdb-missing

// <chrono>
//
// class tzdb_list;
//
// const_iterator erase_after(const_iterator p);
//
// [time.zone.db.list]/5
//   Preconditions: The iterator following p is dereferenceable.
//
// Since there is no Standard way to create a second entry it's not
// possible to fulfill this precondition. This is tested in a libc++
// specific test.

#include <chrono>
#include <concepts>

std::chrono::tzdb_list& list = std::chrono::get_tzdb_list();
static_assert(std::same_as<decltype(list.erase_after(std::chrono::tzdb_list::const_iterator{})),
                           std::chrono::tzdb_list::const_iterator>);
