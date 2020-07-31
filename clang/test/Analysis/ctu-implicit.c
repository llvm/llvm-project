// RUN: rm -rf %t && mkdir %t
// RUN: mkdir -p %t/ctudir2
// RUN: %clang_cc1  \
// RUN:   -emit-pch -o %t/ctudir2/ctu-import.c.ast %S/Inputs/ctu-import.c
// RUN: cp %S/Inputs/ctu-import.c.externalDefMap.ast-dump.txt %t/ctudir2/externalDefMap.txt
// RUN: %clang_cc1 -analyze \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config  display-ctu-progress=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir2 \
// RUN:   -verify %s

void clang_analyzer_eval(int);

int testStaticImplicit(void);
int func(void) {
  int ret = testStaticImplicit();
  clang_analyzer_eval(ret == 4); // expected-warning{{TRUE}}
  return testStaticImplicit();
}
