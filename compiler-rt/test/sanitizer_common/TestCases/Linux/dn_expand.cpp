// RUN: %clangxx %s -o %t -lresolv && %run %t %p

// -lresolv fails on Android.
// UNSUPPORTED: android

#include <assert.h>
#include <resolv.h>
#include <string.h>

#include <sanitizer/msan_interface.h>

void testWrite() {
  char unsigned input[] = {0xff, 0xc5, 0xf7, 0xff, 0x00, 0x00, 0xff, 0x0a, 0x00,
                           0x00, 0x00, 0x01, 0x00, 0x00, 0x02, 0x00, 0x01, 0x00,
                           0x10, 0x01, 0x05, 0x00, 0x01, 0x0a, 0x67, 0x6f, 0x6f,
                           0x67, 0x6c, 0x65, 0x2e, 0x63, 0x6f, 0x6d, 0x00};
  char output[1024];

  int res = dn_expand(input, input + sizeof(input), input + 23, output,
                      sizeof(output));

  assert(res == 12);
  assert(strcmp(output, "google\\.com") == 0);
}

void testWriteZeroLength() {
  char unsigned input[] = {
      0xff, 0xc5, 0xf7, 0xff, 0x00, 0x00, 0xff, 0x0a, 0x00, 0x00, 0x00, 0x01,
      0x00, 0x00, 0x02, 0x00, 0x01, 0x00, 0x10, 0x01, 0x05, 0x00, 0x01, 0x00,
  };
  char output[1024];

  int res = dn_expand(input, input + sizeof(input), input + 23, output,
                      sizeof(output));

  assert(res == 1);
  assert(strcmp(output, "") == 0);
}

int main(int iArgc, const char *szArgv[]) {
  testWrite();
  testWriteZeroLength();

  return 0;
}
