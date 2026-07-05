// RUN: %clangxx -O0 -g %s -o %t
//

// bionic/netdb.cpp is not implemented.
// UNSUPPORTED: android

#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

void test1() {
  struct protoent *ptp = getprotoent();
  assert(ptp && ptp->p_name);
  assert(ptp->p_proto == 0);
  char **aliases = ptp->p_aliases;
  while (aliases) {
    printf("%s\n", *aliases);
    aliases++;
  }
  endprotoent();
}

void test2() {
  struct protoent *ptp = getprotobyname("tcp");
  assert(ptp && ptp->p_name);
  assert(ptp->p_proto == 6);
  char **aliases = ptp->p_aliases;
  while (aliases) {
    printf("%s\n", *aliases);
    aliases++;
  }
  endprotoent();
}

void test3() {
  struct protoent *ptp = getprotobynumber(1);
  assert(ptp && ptp->p_name);
  assert(ptp->p_proto == 1);
  char **aliases = ptp->p_aliases;
  while (aliases) {
    printf("%s\n", *aliases);
    aliases++;
  }
  endprotoent();
}

void test4() {
  setprotoent(1);
  struct protoent *ptp = getprotobynumber(1);

  ptp = getprotobynumber(2);
  assert(ptp && ptp->p_name);
  assert(ptp->p_proto == 2);
  endprotoent();
}

int main(void) {
  printf("protoent\n");

  test1();
  test2();
  test3();
  test4();

  return 0;
}
