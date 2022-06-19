// RUN: %clangxx -O0 -g %s -o %t
//
// REQUIRES: linux, freebsd

#include <inttypes.h>
#include <netdb.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#if defined(__linux__)
#define LOOPBACK "loopback"
#else
#define LOOPBACK "your-net"
#endif

void test1() {
  struct netent *ntp = getnetent();
  assert(ntp && ntp->n_name);
  assert(ntp->n_addrtype == 2);
  assert(ntp->n_net == 127);
  endnetent();
}

void test2() {
  struct netent *ntp = getnetbyname(LOOPBACK);
  assert(ntp && ntp->n_name);
  assert(ntp->n_addrtype == 2);
  assert(ntp->n_net == 127);
  endnetent();
}

void test3() {
  struct netent *lb = getnetbyname(LOOPBACK);
  assert(lb);
  struct netent *ntp = getnetbyaddr(lb->n_net, lb->n_addrtype);
  assert(ntp && ntp->n_name);
  assert(ntp->n_addrtype == 2);
  assert(ntp->n_net == 127);
  endnetent();
}

void test4() {
  setnetent(1);

  struct netent *ntp = getnetent();
  assert(ntp && ntp->n_name);
  assert(ntp->n_addrtype == 2);
  assert(ntp->n_net == 127);
  endnetent();
}

int main(void) {
  printf("netent\n");

  test1();
  test2();
  test3();
  test4();

  return 0;
}
