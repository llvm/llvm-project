//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: target={{.+}}-zos{{.*}}

// Validating the following declaration of wcsnrtombs resides in std::__locale::__ibm namespace.
// size_t wcsnrtombs(char*, const wchar_t**, size_t, size_t, mbstate_t*);

#include <locale>

int main(int, char**) {
  const wchar_t* w_string = L"Hello, World!";
  mbstate_t state;
  char mb_string[20];
  size_t w_chars  = wcslen(w_string);
  size_t mb_chars = 0;

  // Convert the wide-character string to a multibyte string
  mb_chars = std::__locale::__ibm::wcsnrtombs(mb_string, &w_string, w_chars, 13, &state);

  return mb_chars == 13;
}
