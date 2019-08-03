// RUN: %clangxx_msan -O0 -g %s -o %t && %run %t
// RUN: %clangxx_msan -O0 -g -DPOSITIVE %s -o %t && not %run %t 2>&1 | FileCheck %s

#include <assert.h>
#include <iconv.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <errno.h>

int main(void) {
  iconv_t cd = iconv_open("ASCII", "ASCII");
  assert(cd != (iconv_t)-1);

  char inbuf_[100];
  strcpy(inbuf_, "sample text");
  char outbuf_[100];
#if defined(__NetBSD__)
  // Some OSes expect the 2nd argument of iconv(3) to be of type const char **
  const char *inbuf = inbuf_;
#else
  char *inbuf = inbuf_;
#endif
  char *outbuf = outbuf_;
  size_t inbytesleft = strlen(inbuf_);
  size_t outbytesleft = sizeof(outbuf_);

#ifdef POSITIVE
  {
    char u;
    char *volatile p = &u;
    inbuf_[5] = *p;
  }
#endif

  size_t res;
  res = iconv(cd, 0, 0, 0, 0);
  assert(res != (size_t)-1);

  res = iconv(cd, 0, 0, &outbuf, &outbytesleft);
  assert(res != (size_t)-1);

  res = iconv(cd, &inbuf, &inbytesleft, &outbuf, &outbytesleft);
  // CHECK: MemorySanitizer: use-of-uninitialized-value
  // CHECK: #0 {{.*}} in main {{.*}}iconv.cpp:[[@LINE-2]]
  assert(res != (size_t)-1);
  assert(inbytesleft == 0);

  assert(memcmp(inbuf_, outbuf_, strlen(inbuf_)) == 0);

  iconv_close(cd);
  return 0;
}
