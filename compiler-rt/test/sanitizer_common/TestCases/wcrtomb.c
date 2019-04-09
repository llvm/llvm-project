// RUN: %clang %s -o %t && %run %t 2>&1

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

int main(int argc, char **argv) {
  mbstate_t state;
  memset(&state, 0, sizeof(state));

  char buff[10];
  size_t res = wcrtomb(buff, L'a', &state);
  assert(res == 1);
  assert(buff[0] == 'a');

  res = wcrtomb(buff, L'\0', &state);
  assert(res == 1);
  assert(buff[0] == '\0');

  res = wcrtomb(NULL, L'\0', &state);
  assert(res == 1);

  res = wcrtomb(buff, L'a', NULL);
  assert(res == 1);
  assert(buff[0] == 'a');

  res = wcrtomb(buff, L'\0', NULL);
  assert(res == 1);
  assert(buff[0] == '\0');

  res = wcrtomb(NULL, L'\0', NULL);
  assert(res == 1);

  return 0;
}
