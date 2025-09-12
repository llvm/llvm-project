// REQUIRES: !windows
// RUN: %clangxx_asan -O0 -fno-builtin %s -o %t && %run %t

#include <wchar.h>

int main() {
  wchar_t src[] = L"abc";
  wchar_t dst[4] = {0};
  wcsncpy(dst, src, 3);
  return 0;
}


