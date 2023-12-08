// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -analyze-function='baruser(int)' -x c++ \
// RUN:   -verify=nonctu %s

// RUN: rm -rf %t && mkdir %t
// RUN: mkdir -p %t/ctudir
// RUN: %clang_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -emit-pch -o %t/ctudir/ctu-onego-existingdef-other.cpp.ast %S/Inputs/ctu-onego-existingdef-other.cpp
// RUN: cp %S/Inputs/ctu-onego-existingdef-other.cpp.externalDefMap.ast-dump.txt %t/ctudir/externalDefMap.txt

// Existing and equal function definition in both TU. `other` calls `bar` thus
// `bar` will be indirectly imported. During the import we recognize that there
// is an existing definition in the main TU, so we don't create a new Decl.
// Thus, ctu should not bifurcate on the call of `bar` it should directly
// inlinie that as in the case of nonctu.
// Note, we would not get a warning below, if `bar` is conservatively evaluated.
int bar() {
  return 0;
}

//Here we completely supress the CTU work list execution. We should not
//bifurcate on the call of `bar`. (We do not load the foreign AST at all.)
// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -verify=stu %s \
// RUN:   -analyze-function='baruser(int)' -x c++ \
// RUN:   -analyzer-config ctu-max-nodes-pct=0 \
// RUN:   -analyzer-config ctu-max-nodes-min=0

//Here we enable the CTU work list execution. We should not bifurcate on the
//call of `bar`.
// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -verify=ctu %s \
// RUN:   -analyze-function='baruser(int)' -x c++ \
// RUN:   -analyzer-config ctu-max-nodes-pct=100 \
// RUN:   -analyzer-config ctu-max-nodes-min=1000
//Check that the AST file is loaded.
// RUN: %clang_analyze_cc1 -std=c++14 -triple x86_64-pc-linux-gnu \
// RUN:   -analyzer-checker=core,debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -analyzer-config experimental-enable-naive-ctu-analysis=true \
// RUN:   -analyzer-config ctu-dir=%t/ctudir \
// RUN:   -analyze-function='baruser(int)' -x c++ \
// RUN:   -analyzer-config ctu-max-nodes-pct=100 \
// RUN:   -analyzer-config display-ctu-progress=true \
// RUN:   -analyzer-config ctu-max-nodes-min=1000 2>&1 %s | FileCheck %s
// CHECK: CTU loaded AST file

void other(); // Defined in the other TU.

void baruser(int) {
  other();
  int x = bar();
  (void)(1 / x);
  // ctu-warning@-1{{Division by zero}}
  // stu-warning@-2{{Division by zero}}
  // nonctu-warning@-3{{Division by zero}}
}
