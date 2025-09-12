// REQUIRES: !windows
// RUN: %clangxx_asan -O0 -fno-builtin %s -o %t && %run %t

#include <wchar.h>

int main() {
  wchar_t dst[16] = L"ab";
  const wchar_t *src = L"cd";
  wcsncat(dst, src, 1);
  return 0;
}


