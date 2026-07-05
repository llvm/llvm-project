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

void Test() {
  constexpr int __attribute__((vector_size(sizeof(int) * 1))) v1i = {1};
  // CHECK:  |-DeclStmt {{.*}} <line:{{.*}}, col:{{.*}}>
  // CHECK-NEXT:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} v1i '__attribute__((__vector_size__(1 * sizeof(int)))) int const' constexpr cinit
  // CHECK-NEXT:  |   |-value: Vector length=1
  // CHECK-NEXT:  |   | `-element: Int 1

  constexpr int __attribute__((vector_size(sizeof(int) * 4))) v4i = {1, 2, 3, 4};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} v4i '__attribute__((__vector_size__(4 * sizeof(int)))) int const' constexpr cinit
  // CHECK-NEXT:  |   |-value: Vector length=4
  // CHECK-NEXT:  |   | `-elements: Int 1, Int 2, Int 3, Int 4

  constexpr int __attribute__((vector_size(sizeof(int) * 5))) v5i = {1, 2, 3, 4};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} v5i '__attribute__((__vector_size__(5 * sizeof(int)))) int const' constexpr cinit
  // CHECK-NEXT:  |   |-value: Vector length=5
  // CHECK-NEXT:  |   | |-elements: Int 1, Int 2, Int 3, Int 4
  // CHECK-NEXT:  |   | `-element: Int 0

  constexpr int __attribute__((vector_size(sizeof(int) * 8))) v8i = {1, 2, 3, 4};
  // CHECK:  | `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} v8i '__attribute__((__vector_size__(8 * sizeof(int)))) int const' constexpr cinit
  // CHECK-NEXT:  |   |-value: Vector length=8
  // CHECK-NEXT:  |   | |-elements: Int 1, Int 2, Int 3, Int 4
  // CHECK-NEXT:  |   | `-elements: Int 0, Int 0, Int 0, Int 0

  constexpr int __attribute__((vector_size(sizeof(int) * 9))) v9i = {1, 2, 3, 4};
  // CHECK:    `-VarDecl {{.*}} <col:{{.*}}, col:{{.*}}> col:{{.*}} v9i '__attribute__((__vector_size__(9 * sizeof(int)))) int const' constexpr cinit
  // CHECK-NEXT:      |-value: Vector length=9
  // CHECK-NEXT:      | |-elements: Int 1, Int 2, Int 3, Int 4
  // CHECK-NEXT:      | |-elements: Int 0, Int 0, Int 0, Int 0
  // CHECK-NEXT:      | `-element: Int 0
}
