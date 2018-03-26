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

// path& remove_filename()

#include "filesystem_include.hpp"
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "test_iterators.h"
#include "count_new.hpp"
#include "filesystem_test_helper.hpp"


struct RemoveFilenameTestcase {
  const char* value;
  const char* expect;
};

const RemoveFilenameTestcase TestCases[] =
  {
      {"", ""}
    , {"/", ""}
    , {"//", ""}
    , {"///", ""}
    , {"\\", ""}
    , {".", ""}
    , {"..", ""}
    , {"/foo", "/"}
    , {"//foo", ""}
    , {"//foo/", ""}
    , {"//foo///", ""}
    , {"///foo", "/"}
    , {"///foo/", "///foo"}
    , {"/foo/", "/foo"}
    , {"/foo/.", "/foo"}
    , {"/foo/..", "/foo"}
    , {"/foo/////", "/foo"}
    , {"/foo\\\\", "/"}
    , {"/foo//\\/", "/foo//\\"}
    , {"///foo", "/"}
    , {"file.txt", ""}
    , {"bar/../baz/./file.txt", "bar/../baz/."}
  };

int main()
{
  using namespace fs;
  for (auto const & TC : TestCases) {
    path const p_orig(TC.value);
    path p(p_orig);
    assert(p == TC.value);
    path& Ref = (p.remove_filename());
    assert(p == TC.expect);
    assert(&Ref == &p);
    {
      const path parentp = p_orig.parent_path();
      if (parentp == p_orig.root_name()) {

        assert(p.empty());
      } else {
        assert(p == parentp);
      }
    }
  }
}
