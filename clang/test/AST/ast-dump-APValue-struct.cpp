// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu++17 \
// RUN:            -ast-dump %s -ast-dump-filter Test \
// RUN: | FileCheck --check-prefixes=CHECK,CHECK-NOPCH --strict-whitespace --match-full-lines %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu++17 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -triple x86_64-unknown-unknown -Wno-unused-value -std=gnu++17 \
// RUN:           -include-pch %t -ast-dump-all -ast-dump-filter Test /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck --strict-whitespace --match-full-lines %s

struct S0 {
  int i = 0;
  union {
    int j = 0;
  } u0;
};

struct S1 {
  int i = 0;
  union {
    struct {
      int j = 0;
    } s;
  } u1;
};

struct S2 {
  int i = 0;
  union {
    union {
      int j = 0;
    } u;
  } u2;
};

struct S3 {
  int i = 0;
  union {
    union {
      struct {
        int j = 0;
      } j;
    } u;
  } u3;
};

struct S4 : S0 {
  int i = 1, j = 2, k = 3;
  struct {
  } s;
  int a = 4, b = 5, c = 6;
};

struct S5 : S4 {
  int arr0[8] = {1, 2, 3, 4};
  int arr1[8] = {1, 2, 3, 4, 0, 0, 0, 0};
};

void Test() {
  constexpr S0 s0{};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} s0 'const S0' constexpr listinit
  // CHECK-NOPCH-NEXT:  |   |-value: Struct
  // CHECK-NOPCH-NEXT:  |   | `-fields: Int 0, Union .j Int 0

  constexpr S1 s1{};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} s1 'const S1' constexpr listinit
  // CHECK-NOPCH-NEXT:  |   |-value: Struct
  // CHECK-NOPCH-NEXT:  |   | |-field: Int 0
  // CHECK-NOPCH-NEXT:  |   | `-field: Union .s
  // CHECK-NOPCH-NEXT:  |   |   `-Struct
  // CHECK-NOPCH-NEXT:  |   |     `-field: Int 0

  constexpr S2 s2{};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} s2 'const S2' constexpr listinit
  // CHECK-NOPCH-NEXT:  |   |-value: Struct
  // CHECK-NOPCH-NEXT:  |   | `-fields: Int 0, Union .u Union .j Int 0

  constexpr S3 s3{};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} s3 'const S3' constexpr listinit
  // CHECK-NOPCH-NEXT:  |   |-value: Struct
  // CHECK-NOPCH-NEXT:  |   | |-field: Int 0
  // CHECK-NOPCH-NEXT:  |   | `-field: Union .u
  // CHECK-NOPCH-NEXT:  |   |   `-Union .j
  // CHECK-NOPCH-NEXT:  |   |     `-Struct
  // CHECK-NOPCH-NEXT:  |   |       `-field: Int 0

  constexpr S4 s4{};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} s4 'const S4' constexpr listinit
  // CHECK-NOPCH-NEXT:  |   |-value: Struct
  // CHECK-NOPCH-NEXT:  |   | |-base: Struct
  // CHECK-NOPCH-NEXT:  |   | | `-fields: Int 0, Union .j Int 0
  // CHECK-NOPCH-NEXT:  |   | |-fields: Int 1, Int 2, Int 3
  // CHECK-NOPCH-NEXT:  |   | |-field: Struct
  // CHECK-NOPCH-NEXT:  |   | `-fields: Int 4, Int 5, Int 6

  constexpr S5 s5{};
  // CHECK:    `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} s5 'const S5' constexpr listinit
  // CHECK-NOPCH-NEXT:      |-value: Struct
  // CHECK-NOPCH-NEXT:      | |-base: Struct
  // CHECK-NOPCH-NEXT:      | | |-base: Struct
  // CHECK-NOPCH-NEXT:      | | | `-fields: Int 0, Union .j Int 0
  // CHECK-NOPCH-NEXT:      | | |-fields: Int 1, Int 2, Int 3
  // CHECK-NOPCH-NEXT:      | | |-field: Struct
  // CHECK-NOPCH-NEXT:      | | `-fields: Int 4, Int 5, Int 6
  // CHECK-NOPCH-NEXT:      | |-field: Array size=8
  // CHECK-NOPCH-NEXT:      | | |-elements: Int 1, Int 2, Int 3, Int 4
  // CHECK-NOPCH-NEXT:      | | `-filler: 4 x Int 0
  // CHECK-NOPCH-NEXT:      | `-field: Array size=8
  // CHECK-NOPCH-NEXT:      |   |-elements: Int 1, Int 2, Int 3, Int 4
  // CHECK-NOPCH-NEXT:      |   `-elements: Int 0, Int 0, Int 0, Int 0
}
