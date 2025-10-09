//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <filesystem>

// class path

// path& replace_extension(path const& p = path())

#include <filesystem>
#include <cassert>
#include <string>
#include <type_traits>

#include "count_new.h"
#include "test_iterators.h"
namespace fs = std::filesystem;

struct ReplaceExtensionTestcase {
  const char* value;
  const char* expect;
  const char* extension;
};

const ReplaceExtensionTestcase TestCases[] =
  {
      {"", "", ""}
    , {"foo.cpp", "foo", ""}
    , {"foo.cpp", "foo.", "."}
    , {"foo..cpp", "foo..txt", "txt"}
    , {"", ".txt", "txt"}
    , {"", ".txt", ".txt"}
    , {"/foo", "/foo.txt", ".txt"}
    , {"/foo", "/foo.txt", "txt"}
    , {"/foo.cpp", "/foo.txt", ".txt"}
    , {"/foo.cpp", "/foo.txt", "txt"}
  };
const ReplaceExtensionTestcase NoArgCases[] =
  {
      {"", "", ""}
    , {"foo", "foo", ""}
    , {"foo.cpp", "foo", ""}
    , {"foo..cpp", "foo.", ""}
};

int main(int, char**)
{
  using namespace fs;
  for (auto const & TC : TestCases) {
    path p(TC.value);
    assert(p == TC.value);
    path& Ref = (p.replace_extension(TC.extension));
    assert(p == TC.expect);
    assert(&Ref == &p);
  }
  for (auto const& TC : NoArgCases) {
    path p(TC.value);
    assert(p == TC.value);
    path& Ref = (p.replace_extension());
    assert(p == TC.expect);
    assert(&Ref == &p);
  }

  return 0;
}
