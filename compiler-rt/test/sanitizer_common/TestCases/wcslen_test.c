// RUN: %clang %s -O0 -o %t && %run %t 2>&1

#include <assert.h>
#include <wchar.h>

int main(int argc, char **argv) {
  wchar_t x[] = L"Hello World!";
  assert(wcslen(x) == 12);
  return 0;
}
