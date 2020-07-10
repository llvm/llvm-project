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
  int arr[2];
};
union U0 {
  int i;
  float f;
};

struct S1 {
  S0 s0 = {1, 2};
  U0 u0 = {.i = 42};
};

void Test() {
  constexpr int __attribute__((vector_size(sizeof(int) * 5))) arr_v5i[5] = {
      {1, 2, 3, 4, 5},
      {1, 2, 3, 4},
  };
  // CHECK:  | `-VarDecl {{.*}} <line:{{.*}}, line:{{.*}}> line:{{.*}} arr_v5i '__attribute__((__vector_size__(5 * sizeof(int)))) int const[5]' constexpr cinit
  // CHECK-NEXT:  |   |-value: Array size=5
  // CHECK-NEXT:  |   | |-element: Vector length=5
  // CHECK-NEXT:  |   | | |-elements: Int 1, Int 2, Int 3, Int 4
  // CHECK-NEXT:  |   | | `-element: Int 5
  // CHECK-NEXT:  |   | |-element: Vector length=5
  // CHECK-NEXT:  |   | | |-elements: Int 1, Int 2, Int 3, Int 4
  // CHECK-NEXT:  |   | | `-element: Int 0
  // CHECK-NEXT:  |   | `-filler: 3 x Vector length=5
  // CHECK-NEXT:  |   |   |-elements: Int 0, Int 0, Int 0, Int 0
  // CHECK-NEXT:  |   |   `-element: Int 0

  constexpr float arr_f[3][5] = {
      {1, 2, 3, 4, 5},
  };
  // CHECK:  | `-VarDecl {{.*}} <line:{{.*}}, line:{{.*}}> line:{{.*}} arr_f 'float const[3][5]' constexpr cinit
  // CHECK-NEXT:  |   |-value: Array size=3
  // CHECK-NEXT:  |   | |-element: Array size=5
  // CHECK-NEXT:  |   | | |-elements: Float 1.000000e+00, Float 2.000000e+00, Float 3.000000e+00, Float 4.000000e+00
  // CHECK-NEXT:  |   | | `-element: Float 5.000000e+00
  // CHECK-NEXT:  |   | `-filler: 2 x Array size=5
  // CHECK-NEXT:  |   |   `-filler: 5 x Float 0.000000e+00

  constexpr S0 arr_s0[2] = {{1, 2}, {3, 4}};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} arr_s0 'S0 const[2]' constexpr cinit
  // CHECK-NEXT:  |   |-value: Array size=2
  // CHECK-NEXT:  |   | |-element: Struct
  // CHECK-NEXT:  |   | | `-field: Array size=2
  // CHECK-NEXT:  |   | |   `-elements: Int 1, Int 2
  // CHECK-NEXT:  |   | `-element: Struct
  // CHECK-NEXT:  |   |   `-field: Array size=2
  // CHECK-NEXT:  |   |     `-elements: Int 3, Int 4

  constexpr U0 arr_u0[2] = {{.i = 42}, {.f = 3.1415f}};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} arr_u0 'U0 const[2]' constexpr cinit
  // CHECK-NEXT:  |   |-value: Array size=2
  // CHECK-NEXT:  |   | `-elements: Union .i Int 42, Union .f Float 3.141500e+00

  constexpr S1 arr_s1[2] = {};
  // CHECK:    `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} arr_s1 'S1 const[2]' constexpr cinit
  // CHECK-NEXT:      |-value: Array size=2
  // CHECK-NEXT:      | |-element: Struct
  // CHECK-NEXT:      | | |-field: Struct
  // CHECK-NEXT:      | | | `-field: Array size=2
  // CHECK-NEXT:      | | |   `-elements: Int 1, Int 2
  // CHECK-NEXT:      | | `-field: Union .i Int 42
  // CHECK-NEXT:      | `-element: Struct
  // CHECK-NEXT:      |   |-field: Struct
  // CHECK-NEXT:      |   | `-field: Array size=2
  // CHECK-NEXT:      |   |   `-elements: Int 1, Int 2
  // CHECK-NEXT:      |   `-field: Union .i Int 42
}
