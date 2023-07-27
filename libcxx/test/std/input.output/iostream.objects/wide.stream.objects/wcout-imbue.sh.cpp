//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <iostream>

// wostream wcout;

// UNSUPPORTED: no-wide-characters

// UNSUPPORTED: executor-has-no-bash
// FILE_DEPENDENCIES: ../check-stdout.sh
// RUN: %{build}
// RUN: %{exec} bash check-stdout.sh "%t.exe" "zzzz"

#include <iostream>

struct custom_codecvt : std::codecvt<wchar_t, char, std::mbstate_t> {
  using base = std::codecvt<wchar_t, char, std::mbstate_t>;
protected:
  result do_out(std::mbstate_t&, const wchar_t *from, const wchar_t *from_end,
                const wchar_t *&from_next, char *to, char *to_end, char *&to_next) const {
    while (from != from_end && to != to_end) {
      ++from;
      *to++ = 'z';
    }
    from_next = from;
    to_next = to;
    return ok;
  }
};

int main(int, char**) {
    std::locale loc(std::locale::classic(), new custom_codecvt);
    std::wcout.imbue(loc);
    std::wcout << L"1234";
    return 0;
}
