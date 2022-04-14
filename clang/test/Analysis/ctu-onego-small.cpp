// RUN: rm -rf %t && mkdir %t
// RUN: mkdir -p %t/ctudir
// RUN: %clang_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -emit-pch -o %t/ctudir/ctu-onego-small-other.cpp.ast %S/Inputs/ctu-onego-small-other.cpp
// RUN: cp %S/Inputs/ctu-onego-small-other.cpp.externalDefMap.ast-dump.txt %t/ctudir/externalDefMap.txt

// Small function defined in another TU.
int bar();

// Here we limit the ctu analysis to the first phase only (via the
// ctu-max-nodes config options). And we check whether the small foreign
// function `bar` is inlined.

// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -analyzer-config display-ctu-progress=true \
// RUN:   -analyzer-display-progress \
// RUN:   -analyzer-config ctu-max-nodes-pct=0 \
// RUN:   -analyzer-config ctu-max-nodes-min=0 2>&1 %s | FileCheck %s
// CHECK: ANALYZE (Path,  Inline_Regular): {{.*}} baruser(int){{.*}}CTU loaded AST file

// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -analyzer-config ctu-max-nodes-pct=0 \
// RUN:   -analyzer-config ctu-phase1-inlining=none \
// RUN:   -analyzer-config ctu-max-nodes-min=0 -verify=inline-none %s

// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -analyzer-config ctu-max-nodes-pct=0 \
// RUN:   -analyzer-config ctu-phase1-inlining=small \
// RUN:   -analyzer-config ctu-max-nodes-min=0 -verify=inline-small %s


void clang_analyzer_eval(int);

void baruser(int x) {
  int y = bar();
  // inline-none-warning@+2{{UNKNOWN}}
  // inline-small-warning@+1{{TRUE}}
  clang_analyzer_eval(y == 0);
}
