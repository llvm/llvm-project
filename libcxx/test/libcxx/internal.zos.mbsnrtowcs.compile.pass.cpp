//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: target={{.+}}-zos{{.*}}

// Validating the following declaration of mbsnrtowcs resides in std::__locale::__ibm namespace.
// size_t mbsnrtowcs(wchar_t*, const char**, size_t, size_t, mbstate_t*);

#include <locale>

int main(int, char**) {
  const char* mb_string = "Hello, World!";
  wchar_t w_string[20];
  mbstate_t state;
  size_t mb_chars = strlen(mb_string);
  size_t w_chars  = 0;

  // Convert the multibyte string to a wide-character string
  w_chars = std::__locale::__ibm::mbsnrtowcs(w_string, &mb_string, mb_chars, 13, &state);

  return w_chars == 13;
}
