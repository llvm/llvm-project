// RUN: %clangxx_msan -g -O0 %s -o %t && \
// RUN:     %run %t 2>&1
// RUN: %clangxx_msan -g -O0 -DUNINIT=1 %s -o %t && \
// RUN:     not %run %t 2>&1 | FileCheck %s

// XFAIL: target-is-mips64el

#include <assert.h>
#include <string.h>
#include <rpc/xdr.h>

#include <sanitizer/msan_interface.h>

int main(int argc, char *argv[]) {
  XDR xdrs;
  char buf[100];
  xdrmem_create(&xdrs, buf, sizeof(buf), XDR_ENCODE);
  char s[20];
#ifndef UNINIT
  strcpy(s, "hello");
#endif
  char *sp = s;
  bool_t res = xdr_string(&xdrs, &sp, sizeof(s));
  // CHECK: MemorySanitizer: use-of-uninitialized-value
  // CHECK: {{in main.*sunrpc_string.cc:}}[[@LINE-2]]
  assert(res == TRUE);
  xdr_destroy(&xdrs);

  xdrmem_create(&xdrs, buf, sizeof(buf), XDR_DECODE);
  char s2[20];
  char *sp2 = s2;
  res = xdr_string(&xdrs, &sp2, sizeof(s2));
  assert(res == TRUE);
  assert(strcmp(s, s2) == 0);
  xdr_destroy(&xdrs);
  return 0;
}
