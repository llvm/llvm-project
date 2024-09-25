//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test hangs on Android devices that lack shell_v2, which was added in
// Android N (API 24).
// UNSUPPORTED: LIBCXX-ANDROID-FIXME && android-device-api={{2[1-3]}}

// <iostream>

// wistream wcin;

// UNSUPPORTED: no-wide-characters

// RUN: %{build}
// RUN: echo -n 1234 > %t.input
// RUN: %{exec} %t.exe < %t.input

#include <iostream>
#include <cassert>

struct custom_codecvt : std::codecvt<wchar_t, char, std::mbstate_t> {
  using base = std::codecvt<wchar_t, char, std::mbstate_t>;
protected:
  result do_in(std::mbstate_t&, const char *from, const char *from_end,
                const char *&from_next, wchar_t *to, wchar_t *to_end, wchar_t *&to_next) const {
    while (from != from_end && to != to_end) {
      ++from;
      *to++ = L'z';
    }
    from_next = from;
    to_next = to;
    return ok;
  }
};

int main(int, char**) {
    std::locale loc(std::locale::classic(), new custom_codecvt);
    std::wcin.imbue(loc);
    std::wstring str;
    std::wcin >> str;
    assert(str == L"zzzz");
    return 0;
}
