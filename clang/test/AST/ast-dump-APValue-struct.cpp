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
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} s0 'const S0':'const S0' constexpr listinit
  // CHECK-NEXT:  |   |-value: Struct
  // CHECK-NEXT:  |   | `-fields: Int 0, Union .j Int 0

  constexpr S1 s1{};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} s1 'const S1':'const S1' constexpr listinit
  // CHECK-NEXT:  |   |-value: Struct
  // CHECK-NEXT:  |   | |-field: Int 0
  // CHECK-NEXT:  |   | `-field: Union .s
  // CHECK-NEXT:  |   |   `-Struct
  // CHECK-NEXT:  |   |     `-field: Int 0

  constexpr S2 s2{};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} s2 'const S2':'const S2' constexpr listinit
  // CHECK-NEXT:  |   |-value: Struct
  // CHECK-NEXT:  |   | `-fields: Int 0, Union .u Union .j Int 0

  constexpr S3 s3{};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} s3 'const S3':'const S3' constexpr listinit
  // CHECK-NEXT:  |   |-value: Struct
  // CHECK-NEXT:  |   | |-field: Int 0
  // CHECK-NEXT:  |   | `-field: Union .u
  // CHECK-NEXT:  |   |   `-Union .j
  // CHECK-NEXT:  |   |     `-Struct
  // CHECK-NEXT:  |   |       `-field: Int 0

  constexpr S4 s4{};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} s4 'const S4':'const S4' constexpr listinit
  // CHECK-NEXT:  |   |-value: Struct
  // CHECK-NEXT:  |   | |-base: Struct
  // CHECK-NEXT:  |   | | `-fields: Int 0, Union .j Int 0
  // CHECK-NEXT:  |   | |-fields: Int 1, Int 2, Int 3
  // CHECK-NEXT:  |   | |-field: Struct
  // CHECK-NEXT:  |   | `-fields: Int 4, Int 5, Int 6

  constexpr S5 s5{};
  // CHECK:    `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} s5 'const S5':'const S5' constexpr listinit
  // CHECK-NEXT:      |-value: Struct
  // CHECK-NEXT:      | |-base: Struct
  // CHECK-NEXT:      | | |-base: Struct
  // CHECK-NEXT:      | | | `-fields: Int 0, Union .j Int 0
  // CHECK-NEXT:      | | |-fields: Int 1, Int 2, Int 3
  // CHECK-NEXT:      | | |-field: Struct
  // CHECK-NEXT:      | | `-fields: Int 4, Int 5, Int 6
  // CHECK-NEXT:      | |-field: Array size=8
  // CHECK-NEXT:      | | |-elements: Int 1, Int 2, Int 3, Int 4
  // CHECK-NEXT:      | | `-filler: 4 x Int 0
  // CHECK-NEXT:      | `-field: Array size=8
  // CHECK-NEXT:      |   |-elements: Int 1, Int 2, Int 3, Int 4
  // CHECK-NEXT:      |   `-elements: Int 0, Int 0, Int 0, Int 0
}
