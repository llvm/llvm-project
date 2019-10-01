// RUN: %clang %s -o %t && %run %t 2>&1

#include <assert.h>
#include <stdlib.h>

int main(int argc, char **argv) {
  char buff[10];
  wchar_t x = L'a';
  wctomb(NULL, x);
  int res = wctomb(buff, x);
  assert(res == 1);
  assert(buff[0] == 'a');
  return 0;
}
