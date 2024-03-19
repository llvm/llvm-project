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

// <chrono>

//  struct sys_info {
//    sys_seconds   begin;
//    sys_seconds   end;
//    seconds       offset;
//    minutes       save;
//    string        abbrev;
//  };

#include <chrono>
#include <string>

std::chrono::sys_info sys_info;

[[maybe_unused]] std::chrono::sys_seconds& begin = sys_info.begin;
[[maybe_unused]] std::chrono::sys_seconds& end   = sys_info.end;
[[maybe_unused]] std::chrono::seconds& offset    = sys_info.offset;
[[maybe_unused]] std::chrono::minutes& save      = sys_info.save;
[[maybe_unused]] std::string& abbrev             = sys_info.abbrev;
