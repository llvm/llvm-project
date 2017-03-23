// RUN: %clang_cc1 -O2 -fprofile-sample-use=%S/Inputs/pgo-sample-thinlto-summary.prof %s -emit-llvm -o - 2>&1 | FileCheck %s -check-prefix=INLINE
// RUN: %clang_cc1 -O2 -fprofile-sample-use=%S/Inputs/pgo-sample-thinlto-summary.prof %s -emit-llvm -flto=thin -o - 2>&1 | FileCheck %s -check-prefix=NOINLINE
// Checks if hot call is inlined by normal compile, but not inlined by
// thinlto compile.

int baz(int);
int g;

void foo(int n) {
  for (int i = 0; i < n; i++)
    g += baz(i);
}

// INLINE-NOT: call{{.*}}foo
// NOINLINE: call{{.*}}foo
void bar(int n) {
  for (int i = 0; i < n; i++)
    foo(i);
}
