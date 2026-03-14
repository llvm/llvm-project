// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>

#define STRING_OR_NULL(x) ((x) ? (x) : "null")

void test1() {
  struct protoent *ptp = getprotoent();

  printf("%s ", STRING_OR_NULL(ptp->p_name));

  for (char **cp = ptp->p_aliases; *cp != NULL; cp++)
    printf("%s ", STRING_OR_NULL(*cp));

  printf("%d\n", ptp->p_proto);
  endprotoent();
}

void test2() {
  struct protoent *ptp = getprotobyname("icmp");

  printf("%s ", STRING_OR_NULL(ptp->p_name));

  for (char **cp = ptp->p_aliases; *cp != NULL; cp++)
    printf("%s ", STRING_OR_NULL(*cp));

  printf("%d\n", ptp->p_proto);
  endprotoent();
}

void test3() {
  struct protoent *ptp = getprotobynumber(1);

  printf("%s ", STRING_OR_NULL(ptp->p_name));

  for (char **cp = ptp->p_aliases; *cp != NULL; cp++)
    printf("%s ", STRING_OR_NULL(*cp));

  printf("%d\n", ptp->p_proto);
  endprotoent();
}

void test4() {
  setprotoent(1);
  struct protoent *ptp = getprotobynumber(1);

  ptp = getprotobynumber(2);

  printf("%s ", STRING_OR_NULL(ptp->p_name));

  for (char **cp = ptp->p_aliases; *cp != NULL; cp++)
    printf("%s ", STRING_OR_NULL(*cp));

  printf("%d\n", ptp->p_proto);
  endprotoent();
}

void test5() {
  struct protoent *ptp = getprotobyname("ttp");

  printf("%s ", STRING_OR_NULL(ptp->p_name));

  for (char **cp = ptp->p_aliases; *cp != NULL; cp++)
    printf("%s ", STRING_OR_NULL(*cp));

  printf("%d\n", ptp->p_proto);
  endprotoent();
}

int main(void) {
  printf("protoent\n");

  test1();
  test2();
  test3();
  test4();
  test5();

  // CHECK: protoent
  // CHECK: hopopt HOPOPT 0
  // CHECK: icmp ICMP 1
  // CHECK: icmp ICMP 1
  // CHECK: igmp IGMP 2
  // CHECK: ttp TTP iptm IPTM 84

  return 0;
}
