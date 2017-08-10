//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03
// UNSUPPORTED: libcpp-no-exceptions

// <experimental/filesystem>

// class path

#define _LIBCPP_DEBUG 0
#define _LIBCPP_ASSERT(cond, msg) ((cond) ? ((void)0) : throw 42)

#include <experimental/filesystem>
#include <iterator>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "filesystem_test_helper.hpp"

namespace fs = std::experimental::filesystem;

int main() {
  using namespace fs;
  // Test incrementing/decrementing a singular iterator
  {
    path::iterator singular;
    try {
      ++singular;
      assert(false);
    } catch (int) {}
    try {
      --singular;
      assert(false);
    } catch (int) {}
  }
  // Test decrementing the begin iterator
  {
    path p("foo/bar");
    auto it = p.begin();
    try {
      --it;
      assert(false);
    } catch (int) {}
    ++it;
    ++it;
    try {
      ++it;
      assert(false);
    } catch (int) {}
  }
  // Test incrementing the end iterator
  {
    path p("foo/bar");
    auto it = p.end();
    try {
      ++it;
      assert(false);
    } catch (int) {}
    --it;
    --it;
    try {
      --it;
      assert(false);
    } catch (int) {}
  }
}
