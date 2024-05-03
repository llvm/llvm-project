//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <filesystem>

// class directory_entry

// bool operator==(directory_entry const&) const noexcept;
// bool operator!=(directory_entry const&) const noexcept;
// bool operator< (directory_entry const&) const noexcept;
// bool operator<=(directory_entry const&) const noexcept;
// bool operator> (directory_entry const&) const noexcept;
// bool operator>=(directory_entry const&) const noexcept;
// strong_ordering operator<=>(directory_entry const&) const noexcept;

#include <filesystem>
#include <cassert>
#include <type_traits>
#include <utility>

#include "test_macros.h"
#include "test_comparisons.h"
namespace fs = std::filesystem;

int main(int, char**) {
  using namespace fs;

  AssertComparisonsAreNoexcept<directory_entry>();
  AssertComparisonsReturnBool<directory_entry>();
#if TEST_STD_VER > 17
  AssertOrderAreNoexcept<directory_entry>();
  AssertOrderReturn<std::strong_ordering, directory_entry>();
#endif

  typedef std::pair<path, path> TestType;
  TestType TestCases[] = {{"", ""}, {"", "a"}, {"a", "a"}, {"a", "b"}, {"foo/bar/baz", "foo/bar/baz/"}};
  for (auto const& TC : TestCases) {
    assert(testComparisonsValues<directory_entry>(TC.first, TC.second));
    assert(testComparisonsValues<directory_entry>(TC.second, TC.first));
#if TEST_STD_VER > 17
    assert(testOrderValues<directory_entry>(TC.first, TC.second));
    assert(testOrderValues<directory_entry>(TC.second, TC.first));
#endif
  }

  return 0;
}
