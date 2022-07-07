// RUN: %clangxx %s -o %t && %run %t %p

#include <assert.h>
#include <resolv.h>
#include <string.h>

#include "sanitizer_common/sanitizer_specific.h"

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
  check_mem_is_good(output, strlen(output) + 1);
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
  check_mem_is_good(output, strlen(output) + 1);
}

void testComp() {
  char unsigned msg[1024];
  char unsigned *mb = msg;
  char unsigned *me = msg + sizeof(msg);
  char unsigned **pb = (char unsigned **)mb;
  pb[0] = msg;
  pb[1] = nullptr;
  mb += 64;
  char unsigned **pe = (char unsigned **)mb;

  char unsigned *n1 = mb;
  int res = dn_comp("llvm.org", mb, me - mb, pb, pe);
  assert(res == 10);
  mb += res;

  char unsigned *n2 = mb;
  res = dn_comp("lab.llvm.org", mb, me - mb, pb, pe);
  assert(res == 6);
  mb += res;

  {
    char output[1024];
    res = dn_expand(msg, msg + sizeof(msg), n1, output, sizeof(output));

    fprintf(stderr, "%d\n", res);
    assert(res == 10);
    assert(strcmp(output, "llvm.org") == 0);
    check_mem_is_good(output, strlen(output) + 1);
  }

  {
    char output[1024];
    res = dn_expand(msg, msg + sizeof(msg), n2, output, sizeof(output));

    assert(res == 6);
    assert(strcmp(output, "lab.llvm.org") == 0);
    check_mem_is_good(output, strlen(output) + 1);
  }
}

int main(int iArgc, const char *szArgv[]) {
  testWrite();
  testWriteZeroLength();
  testComp();

  return 0;
}
