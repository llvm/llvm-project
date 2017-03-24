// RUN: %clang_cc1 -O2 -fprofile-sample-use=%S/Inputs/pgo-sample-thinlto-summary.prof %s -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=O2
// RUN: %clang_cc1 -O2 -fprofile-sample-use=%S/Inputs/pgo-sample-thinlto-summary.prof %s -emit-llvm -flto=thin -o - 2>&1 | FileCheck %s -check-prefix=THINLTO
// Checks if hot call is inlined by normal compile, but not inlined by
// thinlto compile.

int baz(int);
int g;

void foo(int n) {
  for (int i = 0; i < n; i++)
    g += baz(i);
}

// O2-LABEL: define void @bar
// THINLTO-LABEL: define void @bar
// O2-NOT: call{{.*}}foo
// THINLTO: call{{.*}}foo
void bar(int n) {
  for (int i = 0; i < n; i++)
    foo(i);
}

// Checks if loop unroll is invoked by normal compile, but not thinlto compile.
// O2-LABEL: define void @unroll
// THINLTO-LABEL: define void @unroll
// O2: call{{.*}}baz
// O2: call{{.*}}baz
// THINLTO: call{{.*}}baz
// THINLTO-NOT: call{{.*}}baz
void unroll() {
  for (int i = 0; i < 2; i++)
    baz(i);
}

// Checks if icp is invoked by normal compile, but not thinlto compile.
// O2-LABEL: define void @icp
// THINLTO-LABEL: define void @icp
// O2: if.true.direct_targ
// ThinLTO-NOT: if.true.direct_targ
void icp(void (*p)()) {
  p();
}
