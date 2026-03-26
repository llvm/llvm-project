// RUN: rm -rf %t && mkdir %t
// RUN: mkdir -p %t/ctudir
// RUN: %clang_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -emit-pch -o %t/ctudir/ctu-onego-toplevel-other.cpp.ast %S/Inputs/ctu-onego-toplevel-other.cpp
// RUN: cp %S/Inputs/ctu-onego-toplevel-other.cpp.externalDefMap.ast-dump.txt %t/ctudir/externalDefMap.txt

// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -analyzer-config ctu-phase1-inlining=none \
// RUN:   -verify=ctu %s

// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -analyzer-config ctu-phase1-inlining=none \
// RUN:   -analyzer-config display-ctu-progress=true \
// RUN:   -analyzer-display-progress \
// RUN:   -verify=ctu %s 2>&1 | FileCheck %s

// CallGraph: c->b
// topological sort: c, b
// Note that `other` calls into `b` but that is not visible in the CallGraph
// because that happens in another TU.

// During the onego CTU analysis, we start with c() as top level function.
// Then we visit b() as non-toplevel during the processing of the FWList, thus
// that would not be visited as toplevel without special care.

// `c` is analyzed as toplevel and during that the other TU is loaded:
// CHECK: ANALYZE (Path,  Inline_Regular): {{.*}} c(int){{.*}}CTU loaded AST file
// next, `b` is analyzed as toplevel:
// CHECK: ANALYZE (Path,  Inline_Regular): {{.*}} b(int)

void b(int x);
void other(int y);
void c(int y) {
  other(y);
  return;
  // The below call is here to form the proper CallGraph, but will not be
  // analyzed.
  b(1);
}

void b(int x) {
  if (x == 0)
    (void)(1 / x);
    // ctu-warning@-1{{Division by zero}}
    // We receive the above warning only if `b` is analyzed as top-level.
}
