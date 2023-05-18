// RUN: not %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -ast-dump %s | FileCheck %s

void test() {
  using ContainsErrors = int[sizeof(undef())];
  // CHECK: DependentSizedArrayType {{.*}} contains-errors dependent
}
