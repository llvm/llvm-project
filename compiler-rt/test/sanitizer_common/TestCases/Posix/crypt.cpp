// RUN: %clangxx -O0 -g %s -o %t -lcrypt && %run %t

// crypt() is missing from Android and -lcrypt from darwin.
// UNSUPPORTED: android, darwin

#include <assert.h>
#include <unistd.h>
#include <cstring>
#if __has_include(<crypt.h>)
#include <crypt.h>
#endif

int
main (int argc, char** argv)
{
  {
    char *p = crypt("abcdef", "xz");
    volatile size_t z = strlen(p);
  }
  {
    char *p = crypt("abcdef", "$1$");
    volatile size_t z = strlen(p);
  }
  {
    char *p = crypt("abcdef", "$5$");
    volatile size_t z = strlen(p);
  }
  {
    char *p = crypt("abcdef", "$6$");
    volatile size_t z = strlen(p);
  }
}
