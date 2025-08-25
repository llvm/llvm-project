//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: availability-filesystem-missing

// <filesystem>

// class path

// int compare(path const&) const noexcept;
// int compare(string_type const&) const;
// int compare(value_type const*) const;
//
// bool operator==(path const&, path const&) noexcept;
// bool operator!=(path const&, path const&) noexcept;
// bool operator< (path const&, path const&) noexcept;
// bool operator<=(path const&, path const&) noexcept;
// bool operator> (path const&, path const&) noexcept;
// bool operator>=(path const&, path const&) noexcept;
// strong_ordering operator<=>(path const&, path const&) noexcept;
//
// size_t hash_value(path const&) noexcept;
// template<> struct hash<filesystem::path>;

#include <filesystem>
#include <cassert>
#include <string>
#include <type_traits>
#include <vector>

#include "assert_macros.h"
#include "count_new.h"
#include "test_comparisons.h"
#include "test_iterators.h"
#include "test_macros.h"
namespace fs = std::filesystem;

struct PathCompareTest {
  const char* LHS;
  const char* RHS;
  int expect;
};

#define LONGA                                                                                                          \
  "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" \
  "AAAAAAAA"
#define LONGB                                                                                                          \
  "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB" \
  "BBBBBBBB"
#define LONGC                                                                                                          \
  "CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC" \
  "CCCCCCCC"
#define LONGD                                                                                                          \
  "DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD" \
  "DDDDDDDD"
const PathCompareTest CompareTestCases[] = {
    {"", "", 0},
    {"a", "", 1},
    {"", "a", -1},
    {"a/b/c", "a/b/c", 0},
    {"b/a/c", "a/b/c", 1},
    {"a/b/c", "b/a/c", -1},
    {"a/b", "a/b/c", -1},
    {"a/b/c", "a/b", 1},
    {"a/b/", "a/b/.", -1},
    {"a/b/", "a/b", 1},
    {"a/b//////", "a/b/////.", -1},
    {"a/.././b", "a///..//.////b", 0},
    {"//foo//bar///baz////", "//foo/bar/baz/", 0}, // duplicate separators
    {"///foo/bar", "/foo/bar", 0},                 // "///" is not a root directory
    {"/foo/bar/", "/foo/bar", 1},                  // trailing separator
    {"foo", "/foo", -1}, // if !this->has_root_directory() and p.has_root_directory(), a value less than 0.
    {"/foo", "foo", 1},  //  if this->has_root_directory() and !p.has_root_directory(), a value greater than 0.
#ifdef _WIN32
    {"C:/a", "C:\\a", 0},
#else
    {"C:/a", "C:\\a", -1},
#endif
    {("//" LONGA "////" LONGB "/" LONGC "///" LONGD), ("//" LONGA "/" LONGB "/" LONGC "/" LONGD), 0},
    {(LONGA "/" LONGB "/" LONGC), (LONGA "/" LONGB "/" LONGB), 1}

};
#undef LONGA
#undef LONGB
#undef LONGC
#undef LONGD

static inline int normalize_ret(int ret) { return ret < 0 ? -1 : (ret > 0 ? 1 : 0); }

void test_compare_basic() {
  using namespace fs;
  for (auto const& TC : CompareTestCases) {
    const path p1(TC.LHS);
    const path p2(TC.RHS);
    std::string RHS(TC.RHS);
    const path::string_type R(RHS.begin(), RHS.end());
    const std::basic_string_view<path::value_type> RV(R);
    const path::value_type* Ptr = R.c_str();
    const int E                 = TC.expect;
    {                           // compare(...) functions
      DisableAllocationGuard g; // none of these operations should allocate

      // check runtime results
      int ret1 = normalize_ret(p1.compare(p2));
      int ret2 = normalize_ret(p1.compare(R));
      int ret3 = normalize_ret(p1.compare(Ptr));
      int ret4 = normalize_ret(p1.compare(RV));

      g.release();
      assert(ret1 == ret2);
      assert(ret1 == ret3);
      assert(ret1 == ret4);
      assert(ret1 == E);

      // check signatures
      ASSERT_NOEXCEPT(p1.compare(p2));
    }
    {                           // comparison operators
      DisableAllocationGuard g; // none of these operations should allocate

      // check signatures
      AssertComparisonsAreNoexcept<path>();
      AssertComparisonsReturnBool<path>();
#if TEST_STD_VER > 17
      AssertOrderAreNoexcept<path>();
      AssertOrderReturn<std::strong_ordering, path>();
#endif

      // check comparison results
      assert(testComparisons(p1, p2, /*isEqual*/ E == 0, /*isLess*/ E < 0));
#if TEST_STD_VER > 17
      assert(testOrder(p1, p2, E <=> 0));
#endif
    }
    { // check hash values
      auto h1 = hash_value(p1);
      auto h2 = hash_value(p2);
      assert((h1 == h2) == (p1 == p2));
      // check signature
      ASSERT_SAME_TYPE(std::size_t, decltype(hash_value(p1)));
      ASSERT_NOEXCEPT(hash_value(p1));
    }
    { // check std::hash
      auto h1 = std::hash<fs::path>()(p1);
      auto h2 = std::hash<fs::path>()(p2);
      assert((h1 == h2) == (p1 == p2));
      // check signature
      ASSERT_SAME_TYPE(std::size_t, decltype(std::hash<fs::path>()(p1)));
      ASSERT_NOEXCEPT(std::hash<fs::path>()(p1));
    }
  }
}

int CompareElements(std::vector<std::string> const& LHS, std::vector<std::string> const& RHS) {
  bool IsLess = std::lexicographical_compare(LHS.begin(), LHS.end(), RHS.begin(), RHS.end());
  if (IsLess)
    return -1;

  bool IsGreater = std::lexicographical_compare(RHS.begin(), RHS.end(), LHS.begin(), LHS.end());
  if (IsGreater)
    return 1;

  return 0;
}

void test_compare_elements() {
  struct {
    std::vector<std::string> LHSElements;
    std::vector<std::string> RHSElements;
    int Expect;
  } TestCases[] = {
      {{"a"}, {"a"}, 0},
      {{"a"}, {"b"}, -1},
      {{"b"}, {"a"}, 1},
      {{"a", "b", "c"}, {"a", "b", "c"}, 0},
      {{"a", "b", "c"}, {"a", "b", "d"}, -1},
      {{"a", "b", "d"}, {"a", "b", "c"}, 1},
      {{"a", "b"}, {"a", "b", "c"}, -1},
      {{"a", "b", "c"}, {"a", "b"}, 1},

  };

  auto BuildPath = [](std::vector<std::string> const& Elems) {
    fs::path p;
    for (auto& E : Elems)
      p /= E;
    return p;
  };

  for (auto& TC : TestCases) {
    fs::path LHS        = BuildPath(TC.LHSElements);
    fs::path RHS        = BuildPath(TC.RHSElements);
    const int ExpectCmp = CompareElements(TC.LHSElements, TC.RHSElements);
    assert(ExpectCmp == TC.Expect);
    const int GotCmp = normalize_ret(LHS.compare(RHS));
    assert(GotCmp == TC.Expect);
  }
}

int main(int, char**) {
  test_compare_basic();
  test_compare_elements();

  return 0;
}
