// RUN: %clang_analyze_cc1 -analyzer-checker=unix.cstring.MissingTerminatingZero \
// RUN:   -analyzer-config unix.cstring.MissingTerminatingZero:IgnoreFunctionArgs='strlen 0;strcpy 1' -verify=all,ignore %s
// RUN: %clang_analyze_cc1 -analyzer-checker=unix.cstring.MissingTerminatingZero \
// RUN:   -analyzer-config unix.cstring.MissingTerminatingZero:IgnoreFunctionArgs='strlen 0;strcpy 1' \
// RUN:   -analyzer-config unix.cstring.MissingTerminatingZero:OmitDefaultIgnoreFunctions=true -verify=all,omitdefault %s

#include "Inputs/system-header-simulator.h"

size_t test1(int i) {
  char buf[1] = {1};
  return strlen(buf);
}

void test2(char *dst) {
  char src[1] = {1};
  strcpy(dst, src);
}

int test3() {
  const char buf[1] = {1};
  return execl("path", buf, 4); // all-warning{{String contains no terminating zero}}
}

void test4(char *dst) {
  char src[3] = {1, 2, 3};
  strncpy(dst, src, 3); // omitdefault-warning{{String contains no terminating zero}}
}
