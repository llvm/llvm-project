//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <ctype.h>
#include <wctype.h>
#include <assert.h>
#include <locale>

// This test makes sure that various macro defined on z/OS in
// <ctype.h> and <cwtype.h> which previously were in conflict
// with functions defined in libc++ <locale_base_api.h> still
// work even though they are undefined in <__undef_macros>
// to remove name collisions.

int main(int, char**) {
  setlocale(LC_ALL, "C");
  {
    char upper = 'A';
    char lower = 'a';
    assert(islower(lower));
    assert(tolower(upper) == lower);
    assert(isupper(upper));
    assert(toupper(lower) == upper);
    assert(isdigit('1'));
    assert(isxdigit('b'));
  }

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
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

    assert(iswctype(L'A', wctype("alpha")));
    assert(iswctype(L'1', wctype("digit")));
    assert(iswspace(wspace));
    assert(iswprint(wprint));
    assert(iswcntrl(wcntrl));
    assert(iswupper(wupper));
    assert(iswlower(wlower));
    assert(iswalpha(walpha));
    assert(iswblank(wblank));
    assert(iswdigit(wdigit));
    assert(iswpunct(wpunct));
    assert(iswxdigit(wxdigit));

    assert(static_cast<wint_t>(wlower) == towlower(wupper));
    assert(static_cast<wint_t>(wupper) == towupper(wlower));
  }
#endif

  return 0;
}
