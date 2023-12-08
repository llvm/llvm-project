// RUN: %clang_cc1 -std=c++11 -emit-pch -o %t.a %s
// RUN: %clang_cc1 -std=c++11 -include-pch %t.a %s -ast-print -o - | FileCheck %s

#ifndef HEADER
#define HEADER

inline void test(int i) {
  switch (i) {
    case 1:
      // Notice that the NullStmt has two attributes.
      // CHECK: {{\[\[clang::fallthrough\]\] \[\[clang::fallthrough\]\]}}
      [[clang::fallthrough]] [[clang::fallthrough]];
    case 2:
      break;
  }
}

#else

void foo(void) {
  test(1);
}

#endif
