// RUN: %clang %s -o %t && %run %t 2>&1

// wcsdup internally calls malloc defined in libc library, however
// aix sanitizers can not intercept functions used in shared libraries,
// so the malloc is not intercepted and this case get error for the free:
// AddressSanitizer: attempting free on address which was not malloc()-ed:
//
// UNSUPPORTED: target={{.*aix.*}}

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
