//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-filesystem, no-localization, no-tzdb

// XFAIL: libcpp-has-no-incomplete-tzdb
// XFAIL: availability-tzdb-missing

// <chrono>

// class tzdb_list {
//  public:
//    tzdb_list(const tzdb_list&) = delete;
//    tzdb_list& operator=(const tzdb_list&) = delete;
//
//    ...
//
//  };
//
// [time.zone.db.list]/1
//   The tzdb_list database is a singleton; the unique object of type
//   tzdb_list can be accessed via the get_tzdb_list() function.
////
// This means the class may not have a default constructor.

#include <chrono>
#include <concepts>

static_assert(!std::copyable<std::chrono::tzdb_list>);
static_assert(!std::movable<std::chrono::tzdb_list>);
static_assert(!std::default_initializable<std::chrono::tzdb_list>);
