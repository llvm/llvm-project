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

struct S0 {
  union {
    int i = 42;
  };
};

union U0 {
  union {
    float f = 3.1415f;
  };
};

union U1 {
  union {
    float f;
  };
};

void Test() {
  constexpr S0 s0{};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} s0 'const S0' constexpr listinit
  // CHECK-NEXT:  |   |-value: Struct
  // CHECK-NEXT:  |   | `-field: Union .i Int 42

  constexpr U0 u0a{};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} u0a 'const U0' constexpr listinit
  // CHECK-NEXT:  |   |-value: Union None

  constexpr U0 u0b{3.1415f};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} u0b 'const U0' constexpr listinit
  // CHECK-NEXT:  |   |-value: Union .U0::(anonymous union at {{.*}}) Union .f Float 3.141500e+00

  constexpr U1 u1a{};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} u1a 'const U1' constexpr listinit
  // CHECK-NEXT:  |   |-value: Union .U1::(anonymous union at {{.*}}) Union .f Float 0.000000e+00

  constexpr U1 u1b{3.1415f};
  // CHECK:    `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} u1b 'const U1' constexpr listinit
  // CHECK-NEXT:      |-value: Union .U1::(anonymous union at {{.*}}) Union .f Float 3.141500e+00
}
