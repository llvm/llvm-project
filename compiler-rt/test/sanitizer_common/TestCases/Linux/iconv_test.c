// RUN: %clang %s -o %t && %run %t
// Verify that even if iconv returned -1
// we still treat the initialized part of outbuf as properly initialized.

// UNSUPPORTED: android

#include <iconv.h>
#include <assert.h>
#include <stdio.h>

int main() {
  iconv_t cd = iconv_open("UTF-8", "no");
  assert(cd != (iconv_t)-1);
  char in[11] = {0x7e, 0x7e, 0x5f, 0x53, 0x55, 0x3e,
                 0x99, 0x3c, 0x7e, 0x7e, 0x7e};
  fprintf(stderr, "cd: %p\n", (void*)cd);
  char out[100];
  char *inbuf = &in[0];
  size_t inbytesleft = 11;
  char *outbuf = &out[0];
  size_t outbytesleft = 100;
  int ret = iconv(cd, &inbuf, &inbytesleft, &outbuf, &outbytesleft);
  assert(ret == -1);
  assert(outbuf - &out[0] == 10);
  for (int i = 0; i < 10; i++) {
    if (out[i] == 0x77) return 1;
    fprintf(stderr, "OUT%d 0x%x -- OK\n", i, (unsigned char)out[i]);
  }
  iconv_close(cd);
}

