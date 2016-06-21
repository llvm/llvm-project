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

#include <experimental/filesystem>
#include <chrono>
#include <type_traits>

// system_clock is used because it meets the requirements of TrivialClock,
// and it's resolution and range of system_clock should match the operating
// systems file time type.
typedef std::chrono::system_clock              ExpectedClock;
typedef std::chrono::time_point<ExpectedClock> ExpectedTimePoint;

int main() {
  static_assert(std::is_same<
          std::experimental::filesystem::file_time_type,
          ExpectedTimePoint
      >::value, "");
}
