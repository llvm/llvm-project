//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cwchar>

// UNSUPPORTED: no-wide-characters

// Tests that include ordering does not affect the definition of wcsstr.
// See: https://llvm.org/PR62638

// clang-format off
#include <cwchar>
#include <iosfwd>
// clang-format on

void func() {
  wchar_t* v1;
  const wchar_t* cv2 = L"/";
  v1 = wcsstr(cv2, L"/"); // expected-error {{assigning to 'wchar_t *' from 'const wchar_t *' discards qualifiers}}
}
