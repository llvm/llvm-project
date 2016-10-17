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

// class path

// path& operator/=(path const&)
// path operator/(path const&, path const&)


#define _LIBCPP_DEBUG 0
#define _LIBCPP_ASSERT(x, m) ((x) ? (void)0 : (void)::AssertCount++)
int AssertCount = 0;

#include <experimental/filesystem>
#include <type_traits>
#include <string_view>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.hpp"
#include "filesystem_test_helper.hpp"

namespace fs = std::experimental::filesystem;

int main()
{
  using namespace fs;
  {
    path lhs("//foo");
    path rhs("/bar");
    assert(AssertCount == 0);
    lhs /= rhs;
    assert(AssertCount == 0);
  }
  {
    path lhs("//foo");
    path rhs("/bar");
    assert(AssertCount == 0);
    (void)(lhs / rhs);
    assert(AssertCount == 0);
  }
  {
    path lhs("//foo");
    path rhs("//bar");
    assert(AssertCount == 0);
    lhs /= rhs;
    assert(AssertCount == 1);
    AssertCount = 0;
  }
  {
    path lhs("//foo");
    path rhs("//bar");
    assert(AssertCount == 0);
    (void)(lhs / rhs);
    assert(AssertCount == 1);
  }
  // FIXME The same error is not diagnosed for the append(Source) and
  // append(It, It) overloads.
}
