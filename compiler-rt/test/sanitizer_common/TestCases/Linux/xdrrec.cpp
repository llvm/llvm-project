// RUN: %clangxx -O0 %s -o %t && %run %t | FileCheck %s
// REQUIRES: sunrpc, !android
#include <cassert>
#include <rpc/xdr.h>

int print_msg(char *handle, char *buf, int len) {
  if (len > 0) {
    for (size_t i = 0; i < len; i++) {
      printf("%02x ", (uint8_t)buf[i]);
    }
    printf("\n");
  }
  return len;
}

int main() {
  XDR xdrs;
  xdrs.x_op = XDR_ENCODE;

  xdrrec_create(&xdrs, 0, 0, nullptr, nullptr, print_msg);
  unsigned foo = 42;
  assert(xdr_u_int(&xdrs, &foo));
  assert(xdrrec_endofrecord(&xdrs, /*sendnow*/ true));
  xdr_destroy(&xdrs);
}

// CHECK: 00 00 00 2a
