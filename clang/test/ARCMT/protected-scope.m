// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -fobjc-arc -x objective-c %s.result
// RUN: arcmt-test --args -triple x86_64-apple-darwin10 -fsyntax-only -x objective-c %s > %t
// RUN: diff %t %s.result

#include "Common.h"

void test(id p, int x) {
  int v;
  switch(x) {
  case 0:
    v++;
    id w1 = p;
    id w2 = p;
    break;
  case 1:
    v++;
    id w3 = p;
    break;
  case 2:
  case 3:
    break;
  default:
    break;
  }
}

void test2(int p) {
  switch (p) {
  case 3:;
    NSObject *o = [[NSObject alloc] init];
    [o release];
    break;
  default:
    break;
  }
}
