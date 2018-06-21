//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <experimental/filesystem>

// typedef TrivialClock file_time_type;

#include "filesystem_include.hpp"
#include <chrono>
#include <type_traits>

// system_clock is used because it meets the requirements of TrivialClock,
// and the resolution and range of system_clock should match the operating
// system's file time type.
typedef std::chrono::system_clock              ExpectedClock;
typedef std::chrono::time_point<ExpectedClock> ExpectedTimePoint;

int main() {
  static_assert(std::is_same<
          fs::file_time_type,
          ExpectedTimePoint
      >::value, "");
}
