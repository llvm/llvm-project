// RUN: %clang %s -o %t && %run %t 2>&1 | FileCheck %s

#include <ctype.h>
#include <limits.h>
#include <locale.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

void check_ctype(void) {
  unsigned char c;
  volatile size_t i = 0; /* a dummy variable to prevent optimizing code out */

  for (c = 0; c < UCHAR_MAX; c++)
    i += !!isalpha(c);
  for (c = 0; c < UCHAR_MAX; c++)
    i += !!isascii(c);
  for (c = 0; c < UCHAR_MAX; c++)
    i += !!isblank(c);
  for (c = 0; c < UCHAR_MAX; c++)
    i += !!iscntrl(c);
  for (c = 0; c < UCHAR_MAX; c++)
    i += !!isdigit(c);
  for (c = 0; c < UCHAR_MAX; c++)
    i += !!isgraph(c);
  for (c = 0; c < UCHAR_MAX; c++)
    i += !!islower(c);
  for (c = 0; c < UCHAR_MAX; c++)
    i += !!isprint(c);
  for (c = 0; c < UCHAR_MAX; c++)
    i += !!ispunct(c);
  for (c = 0; c < UCHAR_MAX; c++)
    i += !!isspace(c);
  for (c = 0; c < UCHAR_MAX; c++)
    i += !!isupper(c);
  for (c = 0; c < UCHAR_MAX; c++)
    i += !!isxdigit(c);
  for (c = 0; c < UCHAR_MAX; c++)
    i += !!isalnum(c);

  for (c = 0; c < UCHAR_MAX; c++)
    i += !!tolower(c);
  for (c = 0; c < UCHAR_MAX; c++)
    i += !!toupper(c);

  i += !!isalpha(EOF);
  i += !!isascii(EOF);
  i += !!isblank(EOF);
  i += !!iscntrl(EOF);
  i += !!isdigit(EOF);
  i += !!isgraph(EOF);
  i += !!islower(EOF);
  i += !!isprint(EOF);
  i += !!ispunct(EOF);
  i += !!isspace(EOF);
  i += !!isupper(EOF);
  i += !!isxdigit(EOF);
  i += !!isalnum(EOF);

  i += !!tolower(EOF);
  i += !!toupper(EOF);

  if (i)
    return;
  else
    return;
}

int main(int argc, char **argv) {
  check_ctype();

  setlocale(LC_ALL, "");

  check_ctype();

  setlocale(LC_ALL, "en_US.UTF-8");

  check_ctype();

  setlocale(LC_CTYPE, "pl_PL.UTF-8");

  check_ctype();

  printf("OK\n");

  // CHECK: OK

  return 0;
}
