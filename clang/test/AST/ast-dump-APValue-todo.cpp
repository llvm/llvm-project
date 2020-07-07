// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu++17 \
// RUN:            -ast-dump %s -ast-dump-filter Test \
// RUN: | FileCheck --strict-whitespace --match-full-lines %s

int i;
struct S {
  int i;
};

void Test() {
  constexpr int *pi = &i;
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} pi 'int *const' constexpr cinit
  // CHECK-NEXT:  |   |-value: LValue <todo>

  constexpr int(S::*pmi) = &S::i;
  // CHECK:    `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} pmi 'int (S::*const)' constexpr cinit
  // CHECK-NEXT:      |-value: MemberPointer <todo>
}
