// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

#include <inttypes.h>
#include <netdb.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define STRING_OR_NULL(x) ((x) ? (x) : "null")

void test1() {
  struct netent *ntp = getnetent();

  printf("%s ", ntp->n_name);

  for (char **cp = ntp->n_aliases; *cp != NULL; cp++)
    printf("%s ", STRING_OR_NULL(*cp));

  printf("%d ", ntp->n_addrtype);
  printf("%" PRIu32 "\n", ntp->n_net);

  endnetent();
}

void test2() {
  struct netent *ntp = getnetbyname("loopback");

  printf("%s ", ntp->n_name);

  for (char **cp = ntp->n_aliases; *cp != NULL; cp++)
    printf("%s ", STRING_OR_NULL(*cp));

  printf("%d ", ntp->n_addrtype);
  printf("%" PRIu32 "\n", ntp->n_net);

  endnetent();
}

void test3() {
  struct netent *ntp = getnetbyaddr(127, 2);

  printf("%s ", ntp->n_name);

  for (char **cp = ntp->n_aliases; *cp != NULL; cp++)
    printf("%s ", STRING_OR_NULL(*cp));

  printf("%d ", ntp->n_addrtype);
  printf("%" PRIu32 "\n", ntp->n_net);

  endnetent();
}

void test4() {
  setnetent(1);

  struct netent *ntp = getnetent();

  printf("%s ", ntp->n_name);

  for (char **cp = ntp->n_aliases; *cp != NULL; cp++)
    printf("%s ", STRING_OR_NULL(*cp));

  printf("%d ", ntp->n_addrtype);
  printf("%" PRIu32 "\n", ntp->n_net);

  endnetent();
}

int main(void) {
  printf("netent\n");

  test1();
  test2();
  test3();
  test4();

  // CHECK: netent
  // CHECK: loopback 2 127
  // CHECK: loopback 2 127
  // CHECK: loopback 2 127
  // CHECK: loopback 2 127

  return 0;
}
