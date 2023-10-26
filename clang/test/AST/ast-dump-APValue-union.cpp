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

union U0 {
  int i = 42;
  float f;
};

union U1 {
  union Uinner {
    int i;
    float f = 3.1415f;
  } uinner;
};

union U2 {
  union Uinner {
    double d;
    int arr[2] = {1, 2};
  } uinner;
};

union U3 {
  union Uinner {
    double d = 3.1415;
    int arr[2];
  } uinner;
  float f;
};

void Test() {
  constexpr U0 u0{};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} u0 'const U0' constexpr listinit
  // CHECK-NEXT:  |   |-value: Union .i Int 42

  constexpr U1 u1{};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} u1 'const U1' constexpr listinit
  // CHECK-NEXT:  |   |-value: Union .uinner Union .f Float 3.141500e+00

  constexpr U2 u2{};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} u2 'const U2' constexpr listinit
  // CHECK-NEXT:  |   |-value: Union .uinner
  // CHECK-NEXT:  |   | `-Union .arr
  // CHECK-NEXT:  |   |   `-Array size=2
  // CHECK-NEXT:  |   |     `-elements: Int 1, Int 2

  constexpr U3 u3a = {.f = 3.1415};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} u3a 'const U3' constexpr cinit
  // CHECK-NEXT:  |   |-value: Union .f Float 3.141500e+00

  constexpr U3 u3b = {.uinner = {}};
  // CHECK:    `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} u3b 'const U3' constexpr cinit
  // CHECK-NEXT:      |-value: Union .uinner Union .d Float 3.141500e+00
}
