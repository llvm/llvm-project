// RUN: %clangxx_msan -g -O0 -DTYPE=int -DFN=xdr_int %s -o %t && \
// RUN:     %run %t 2>&1
// RUN: %clangxx_msan -g -O0 -DTYPE=int -DFN=xdr_int -DUNINIT=1 %s -o %t && \
// RUN:     not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_msan -g -O0 -DTYPE=double -DFN=xdr_double %s -o %t && \
// RUN:     %run %t 2>&1
// RUN: %clangxx_msan -g -O0 -DTYPE=double -DFN=xdr_double -DUNINIT=1 %s -o %t && \
// RUN:     not %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_msan -g -O0 -DTYPE=u_quad_t -DFN=xdr_u_longlong_t %s -o %t && \
// RUN:     %run %t 2>&1
// RUN: %clangxx_msan -g -O0 -DTYPE=u_quad_t -DFN=xdr_u_longlong_t -DUNINIT=1 %s -o %t && \
// RUN:     not %run %t 2>&1 | FileCheck %s

// XFAIL: target-is-mips64el

#include <assert.h>
#include <rpc/xdr.h>

#include <sanitizer/msan_interface.h>

int main(int argc, char *argv[]) {
  XDR xdrs;
  char buf[100];
  xdrmem_create(&xdrs, buf, sizeof(buf), XDR_ENCODE);
  TYPE x;
#ifndef UNINIT
  x = 42;
#endif
  bool_t res = FN(&xdrs, &x);
  // CHECK: MemorySanitizer: use-of-uninitialized-value
  // CHECK: {{in main.*sunrpc.cc:}}[[@LINE-2]]
  assert(res == TRUE);
  xdr_destroy(&xdrs);

  xdrmem_create(&xdrs, buf, sizeof(buf), XDR_DECODE);
  TYPE y;
  res = FN(&xdrs, &y);
  assert(res == TRUE);
  assert(__msan_test_shadow(&y, sizeof(y)) == -1);
  xdr_destroy(&xdrs);
  return 0;
}
