//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

#include <string>
#include <cassert>

#include "test_macros.h"

#ifndef TEST_HAS_NO_CHAR8_T
typedef std::u8string u8string;
#else
typedef std::string u8string;
#endif

int main(int, char**) {
  {
    using namespace std::literals::string_literals;

    ASSERT_SAME_TYPE(decltype("Hi"s), std::string);
    ASSERT_SAME_TYPE(decltype(u8"Hi"s), u8string);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    ASSERT_SAME_TYPE(decltype(L"Hi"s), std::wstring);
#endif
    ASSERT_SAME_TYPE(decltype(u"Hi"s), std::u16string);
    ASSERT_SAME_TYPE(decltype(U"Hi"s), std::u32string);

    std::string foo;
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    std::wstring Lfoo;
#endif
    u8string u8foo;
    std::u16string ufoo;
    std::u32string Ufoo;

    foo = ""s;
    assert(foo.size() == 0);
    u8foo = u8""s;
    assert(u8foo.size() == 0);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    Lfoo = L""s;
    assert(Lfoo.size() == 0);
#endif
    ufoo = u""s;
    assert(ufoo.size() == 0);
    Ufoo = U""s;
    assert(Ufoo.size() == 0);

    foo = " "s;
    assert(foo.size() == 1);
    u8foo = u8" "s;
    assert(u8foo.size() == 1);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    Lfoo = L" "s;
    assert(Lfoo.size() == 1);
#endif
    ufoo = u" "s;
    assert(ufoo.size() == 1);
    Ufoo = U" "s;
    assert(Ufoo.size() == 1);

    foo = "ABC"s;
    assert(foo == "ABC");
    assert(foo == std::string("ABC"));
    u8foo = u8"ABC"s;
    assert(u8foo == u8"ABC");
    assert(u8foo == u8string(u8"ABC"));
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
    Lfoo = L"ABC"s;
    assert(Lfoo == L"ABC");
    assert(Lfoo == std::wstring(L"ABC"));
#endif
    ufoo = u"ABC"s;
    assert(ufoo == u"ABC");
    assert(ufoo == std::u16string(u"ABC"));
    Ufoo = U"ABC"s;
    assert(Ufoo == U"ABC");
    assert(Ufoo == std::u32string(U"ABC"));
  }
  {
    using namespace std::literals;
    std::string foo = ""s;
    assert(foo == std::string());
  }
  {
    using namespace std;
    std::string foo = ""s;
    assert(foo == std::string());
  }

  return 0;
}
