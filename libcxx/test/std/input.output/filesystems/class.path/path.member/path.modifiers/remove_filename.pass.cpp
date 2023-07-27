//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: availability-filesystem-missing

// <filesystem>

// class path

// path& remove_filename()

#include "filesystem_include.h"
#include <type_traits>
#include <cassert>

#include "test_iterators.h"
#include "count_new.h"

struct RemoveFilenameTestcase {
  const char* value;
  const char* expect;
};

const RemoveFilenameTestcase TestCases[] =
  {
      {"", ""}
    , {"/", "/"}
    , {"//", "//"}
    , {"///", "///"}
#ifdef _WIN32
    , {"\\", "\\"}
#else
    , {"\\", ""}
#endif
    , {".", ""}
    , {"..", ""}
    , {"/foo", "/"}
    , {"foo/bar", "foo/"}
    , {"foo/", "foo/"}
#ifdef _WIN32
    , {"//foo", "//foo"}
#else
    , {"//foo", "//"}
#endif
    , {"//foo/", "//foo/"}
    , {"//foo///", "//foo///"}
    , {"///foo", "///"}
    , {"///foo/", "///foo/"}
    , {"/foo/", "/foo/"}
    , {"/foo/.", "/foo/"}
    , {"/foo/..", "/foo/"}
    , {"/foo/////", "/foo/////"}
#ifdef _WIN32
    , {"/foo\\\\", "/foo\\\\"}
#else
    , {"/foo\\\\", "/"}
#endif
    , {"/foo//\\/", "/foo//\\/"}
    , {"///foo", "///"}
    , {"file.txt", ""}
    , {"bar/../baz/./file.txt", "bar/../baz/./"}
  };

int main(int, char**)
{
  using namespace fs;
  for (auto const & TC : TestCases) {
    path const p_orig(TC.value);
    path p(p_orig);
    assert(p == TC.value);
    path& Ref = (p.remove_filename());
    assert(p == TC.expect);
    assert(&Ref == &p);
    assert(!p.has_filename());
  }

  return 0;
}
