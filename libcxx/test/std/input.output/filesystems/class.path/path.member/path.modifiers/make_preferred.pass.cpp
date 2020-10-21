//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// class path

// path& make_preferred()

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.h"
#include "filesystem_test_helper.h"


struct MakePreferredTestcase {
  const char* value;
  const char* expected_posix;
  const char* expected_windows;
};

const MakePreferredTestcase TestCases[] =
  {
      {"", "", ""}
    , {"hello_world", "hello_world", "hello_world"}
    , {"/", "/", "\\"}
    , {"/foo/bar/baz/", "/foo/bar/baz/", "\\foo\\bar\\baz\\"}
    , {"\\", "\\", "\\"}
    , {"\\foo\\bar\\baz\\", "\\foo\\bar\\baz\\", "\\foo\\bar\\baz\\"}
    , {"\\foo\\/bar\\/baz\\", "\\foo\\/bar\\/baz\\", "\\foo\\\\bar\\\\baz\\"}
  };

int main(int, char**)
{
  // This operation is an identity operation on linux.
  // On windows, compare with preferred_win, if set.
  using namespace fs;
  for (auto const & TC : TestCases) {
    path p(TC.value);
    assert(p == TC.value);
    path& Ref = (p.make_preferred());
#ifdef _WIN32
    std::string s(TC.expected_windows);
#else
    std::string s(TC.expected_posix);
#endif
    assert(p.string() == s);
    assert(&Ref == &p);
  }

  return 0;
}
