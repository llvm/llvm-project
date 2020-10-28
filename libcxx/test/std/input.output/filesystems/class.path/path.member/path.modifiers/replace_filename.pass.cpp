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

// path& replace_filename()

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.h"
#include "filesystem_test_helper.h"

struct ReplaceFilenameTestcase {
  const char* value;
  const char* expect;
  const char* filename;
};

const ReplaceFilenameTestcase TestCases[] =
  {
      {"/foo", "/bar", "bar"}
    , {"/foo", "/", ""}
    , {"foo", "bar", "bar"}
    , {"/", "/bar", "bar"}
    , {"\\", "bar", "bar"}
    , {"///", "///bar", "bar"}
    , {"\\\\", "bar", "bar"}
    , {"\\/\\", "\\/bar", "bar"}
    , {".", "bar", "bar"}
    , {"..", "bar", "bar"}
    , {"/foo\\baz/bong/", "/foo\\baz/bong/bar", "bar"}
    , {"/foo\\baz/bong", "/foo\\baz/bar", "bar"}
  };

int main(int, char**)
{
  using namespace fs;
  for (auto const & TC : TestCases) {
    path p(TC.value);
    assert(p == TC.value);
    path& Ref = p.replace_filename(TC.filename);
    assert(p == TC.expect);
    assert(&Ref == &p);
    // Tests Effects "as-if": remove_filename() append(filename)
    {
      path p2(TC.value);
      path replace(TC.filename);
      p2.remove_filename();
      p2 /= replace;
      assert(p == p2);
    }
  }

  return 0;
}
