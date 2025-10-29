//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <locale>
#include <cctype>
#include <cwctype>
#include <cassert>

#include "test_macros.h"

// This test makes sure that various macro defined on z/OS in
// <ctype.h> and <cwtype.h> which previously were in conflict
// with functions defined in libc++ <locale_base_api.h> still
// work even though they are undefined in <__undef_macros>
// to remove name collisions.

int main(int, char**) {
  std::locale loc("C");
  {
    char upper  = 'A';
    char lower  = 'a';
    char digit  = '1';
    char xdigit = 'b';
    auto& CF    = std::use_facet<std::ctype_byname<char> >(loc);
    assert(CF.is(std::ctype_base::lower, lower));
    assert(CF.is(std::ctype_base::upper, upper));
    assert(CF.is(std::ctype_base::digit, digit));
    assert(CF.is(std::ctype_base::xdigit, xdigit));
    assert(lower == CF.tolower(upper));
    assert(upper == CF.toupper(lower));
  }

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    wchar_t wctype  = L'A';
    wchar_t wspace  = L' ';
    wchar_t wprint  = L'^';
    wchar_t wcntrl  = L'';
    wchar_t wupper  = L'A';
    wchar_t wlower  = L'a';
    wchar_t walpha  = L'z';
    wchar_t wblank  = L' ';
    wchar_t wdigit  = L'1';
    wchar_t wpunct  = L',';
    wchar_t wxdigit = L'B';

    assert(std::iswctype(wctype, std::wctype("alpha")));
    assert(std::iswctype(wdigit, std::wctype("digit")));
    assert(std::iswspace(wspace));
    assert(std::iswprint(wprint));
    assert(std::iswcntrl(wcntrl));
    assert(std::iswupper(wupper));
    assert(std::iswlower(wlower));
    assert(std::iswalpha(walpha));
    assert(std::iswblank(wblank));
    assert(std::iswdigit(wdigit));
    assert(std::iswpunct(wpunct));
    assert(std::iswxdigit(wxdigit));

    auto& WF = std::use_facet<std::ctype_byname<wchar_t> >(loc);
    assert(wlower == WF.tolower(wupper));
    assert(wupper == WF.toupper(wlower));
  }
#endif

  return 0;
}
