// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu++17 \
// RUN:            -ast-dump %s -ast-dump-filter Test \
// RUN: | FileCheck --strict-whitespace --match-full-lines %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu++17 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu++17 \
// RUN:           -include-pch %t -ast-dump-all -ast-dump-filter Test /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace --match-full-lines %s

int i;
struct S {
  int i;
};

void Test() {
  constexpr int(S::*pmi) = &S::i;
  // CHECK:    `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} pmi 'int (S::*const)' constexpr cinit
  // CHECK-NEXT:      |-value: MemberPointer <todo>
}
