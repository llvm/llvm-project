//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

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
};

const MakePreferredTestcase TestCases[] =
  {
      {""}
    , {"hello_world"}
    , {"/"}
    , {"/foo/bar/baz/"}
    , {"\\"}
    , {"\\foo\\bar\\baz\\"}
    , {"\\foo\\/bar\\/baz\\"}
  };

int main(int, char**)
{
  // This operation is an identity operation on linux.
  using namespace fs;
  for (auto const & TC : TestCases) {
    path p(TC.value);
    assert(p == TC.value);
    path& Ref = (p.make_preferred());
    assert(p.native() == TC.value);
    assert(&Ref == &p);
  }

  return 0;
}
