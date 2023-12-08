// RUN: rm -rf %t && mkdir %t
// RUN: mkdir -p %t/ctudir
// RUN: %clang_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -emit-pch -o %t/ctudir/ctu-onego-indirect-other.cpp.ast %S/Inputs/ctu-onego-indirect-other.cpp
// RUN: cp %S/Inputs/ctu-onego-indirect-other.cpp.externalDefMap.ast-dump.txt %t/ctudir/externalDefMap.txt

int bar();

// Here we have a foreign function `bar` that is imported when we analyze
// `adirectbaruser`. During the subsequent toplevel analysis of `baruser` we
// should bifurcate on the call of `bar`.

//Ensure the order of the toplevel analyzed functions.
// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -analyzer-display-progress \
// RUN:   -analyzer-inlining-mode=all \
// RUN:   -analyzer-config ctu-phase1-inlining=none \
// RUN:   -analyzer-config ctu-max-nodes-pct=100 \
// RUN:   -analyzer-config ctu-max-nodes-min=1000 2>&1 %s | FileCheck %s
// CHECK: ANALYZE (Path,  Inline_Regular):{{.*}}adirectbaruser(int)
// CHECK: ANALYZE (Path,  Inline_Regular):{{.*}}baruser(int)

// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -analyzer-display-progress \
// RUN:   -analyzer-inlining-mode=all \
// RUN:   -analyzer-config ctu-phase1-inlining=none \
// RUN:   -verify %s \
// RUN:   -analyzer-config ctu-max-nodes-pct=100 \
// RUN:   -analyzer-config ctu-max-nodes-min=1000


void other(); // Defined in the other TU.

void clang_analyzer_eval(int);

void baruser(int x) {
  if (x == 1)
    return;
  int y = bar();
  clang_analyzer_eval(y == 0); // expected-warning{{TRUE}}
                               // expected-warning@-1{{UNKNOWN}}
  other();
}

void adirectbaruser(int) {
  int y = bar();
  (void)y;
  baruser(1);
}

