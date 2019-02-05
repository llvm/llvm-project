//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// test type_info

#include <typeinfo>
#include <string>
#include <cstring>
#include <cassert>

bool test_constructor_explicit(std::type_info const&) { return false; }
bool test_constructor_explicit(std::string const&) { return true; }

int main()
{
  {
    const std::type_info& t1 = typeid(int);
    const std::type_info& t2 = typeid(int);
    assert(t1 == t2);
    const std::type_info& t3 = typeid(short);
    assert(t1 != t3);
    assert(!t1.before(t2));
    assert(std::strcmp(t1.name(), t2.name()) == 0);
    assert(std::strcmp(t1.name(), t3.name()) != 0);
  }
  {
    // type_info has a protected constructor taking a string literal. This
    // constructor is not intended for users. However it still participates
    // in overload resolution, so we need to ensure that it is marked explicit
    // to avoid ambiguous conversions.
    // See: https://bugs.freebsd.org/bugzilla/show_bug.cgi?id=216201
    assert(test_constructor_explicit("abc"));
  }
}
