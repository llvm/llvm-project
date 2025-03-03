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
  int ii;
};
S s;

struct F {
  char padding[12];
  S s;
};
F f;

void Test(int (&arr)[10]) {
  constexpr int *pi = &i;
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} pi 'int *const' constexpr cinit
  // CHECK-NEXT:  |   |-value: LValue Base=VarDecl {{.*}}, Null=0, Offset=0, HasPath=1, PathLength=0, Path=()

  constexpr int *psi = &s.i;
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} psi 'int *const' constexpr cinit
  // CHECK-NEXT:  |   |-value: LValue Base=VarDecl {{.*}}, Null=0, Offset=0, HasPath=1, PathLength=1, Path=({{.*}})

  constexpr int *psii = &s.ii;
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} psii 'int *const' constexpr cinit
  // CHECK-NEXT:  |   |-value: LValue Base=VarDecl {{.*}}, Null=0, Offset=4, HasPath=1, PathLength=1, Path=({{.*}})

  constexpr int *pf = &f.s.ii;
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} pf 'int *const' constexpr cinit
  // CHECK-NEXT:  |   |-value: LValue Base=VarDecl {{.*}}, Null=0, Offset=16, HasPath=1, PathLength=2, Path=({{.*}}, {{.*}})

  constexpr char *pc = &f.padding[2];
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} pc 'char *const' constexpr cinit
  // CHECK-NEXT:  |   |-value: LValue Base=VarDecl {{.*}}, Null=0, Offset=2, HasPath=1, PathLength=2, Path=({{.*}}, 2)

  constexpr const int *n = nullptr;
  // CHECK:    `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} n 'const int *const' constexpr cinit
  // CHECK-NEXT:      |-value: LValue Base=null, Null=1, Offset=0, HasPath=1, PathLength=0, Path=()
}
