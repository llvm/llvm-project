// RUN: %clang %s -o %t && %run %t 2>&1

#include <assert.h>
#include <stdlib.h>
#include <wchar.h>

int main(int argc, char **argv) {
  wchar_t *buff = wcsdup(L"foo");
  assert(buff[0] == L'f');
  assert(buff[1] == L'o');
  assert(buff[2] == L'o');
  assert(buff[3] == L'\0');
  free(buff);
  return 0;
}
