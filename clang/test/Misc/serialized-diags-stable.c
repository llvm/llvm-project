// RUN: rm -f %t
// RUN: not %clang -Wall -fsyntax-only %s --serialize-diagnostics %t.dia > /dev/null 2>&1
// RUN: c-index-test -read-diagnostics %t.dia 2>&1 | FileCheck %s

// RUN: c-index-test -read-diagnostics %S/Inputs/serialized-diags-stable.dia 2>&1 | FileCheck %s

int foo() {
  // CHECK: serialized-diags-stable.c:[[@LINE+2]]:1: warning: non-void function does not return a value [-Wreturn-type] [Semantic Issue]
  // CHECK-NEXT: Number FIXITs = 0
}

// CHECK: serialized-diags-stable.c:[[@LINE+5]]:13: error: redefinition of 'bar' as different kind of symbol [] [Semantic Issue]
// CHECK-NEXT: Number FIXITs = 0
// CHECK-NEXT: +-{{.*}}serialized-diags-stable.c:[[@LINE+2]]:6: note: previous definition is here [] []
// CHECK-NEXT: Number FIXITs = 0
void bar() {}
typedef int bar;


// CHECK-LABEL: Number of diagnostics: 2
