// RUN: %clang_cc1 -std=c++20 -fexperimental-new-constant-interpreter -ast-dump %s | FileCheck %s
// RUN: %clang_cc1 -std=c++20                                         -ast-dump %s | FileCheck %s

/// The APValue of arrays with a zero filler uses an array filler.

struct S { int a; int b; };

constexpr int GlobalInts[100] = {1, 2};
// CHECK:      VarDecl {{.*}} GlobalInts 'const int[100]'
// CHECK-NEXT:   value: Array size=100
// CHECK-NEXT:     elements: Int 1, Int 2
// CHECK-NEXT:     filler: 98 x Int 0

constexpr char GlobalStr[10] = "abc";
// CHECK:      VarDecl {{.*}} GlobalStr 'const char[10]'
// CHECK-NEXT:   value: Array size=10
// CHECK-NEXT:     elements: Int 97, Int 98, Int 99
// CHECK-NEXT:     filler: 7 x Int 0

constexpr char GlobalStr2[10] = {"abc"};
// CHECK:      VarDecl {{.*}} GlobalStr2 'const char[10]'
// CHECK-NEXT:   value: Array size=10
// CHECK-NEXT:     elements: Int 97, Int 98, Int 99
// CHECK-NEXT:     filler: 7 x Int 0

constexpr S GlobalRecords[10] = {{1, 2}};
// CHECK:      VarDecl {{.*}} GlobalRecords 'const S[10]'
// CHECK-NEXT:   value: Array size=10
// CHECK-NEXT:     element: Struct
// CHECK-NEXT:       fields: Int 1, Int 2
// CHECK-NEXT:     filler: 9 x Struct
// CHECK-NEXT:       fields: Int 0, Int 0

constexpr int GlobalNoFiller[3] = {1, 2, 3};
// CHECK:      VarDecl {{.*}} GlobalNoFiller 'const int[3]'
// CHECK-NEXT:   value: Array size=3
// CHECK-NEXT:     elements: Int 1, Int 2, Int 3
// CHECK-NEXT:   InitListExpr

void func() {
  constexpr int LocalInts[100] = {1, 2};
  constexpr S LocalRecords[10] = {{1, 2}};
}
// CHECK:      VarDecl {{.*}} LocalInts 'const int[100]'
// CHECK-NEXT:   value: Array size=100
// CHECK-NEXT:     elements: Int 1, Int 2
// CHECK-NEXT:     filler: 98 x Int 0

// CHECK:      VarDecl {{.*}} LocalRecords 'const S[10]'
// CHECK-NEXT:   value: Array size=10
// CHECK-NEXT:     element: Struct
// CHECK-NEXT:       fields: Int 1, Int 2
// CHECK-NEXT:     filler: 9 x Struct
// CHECK-NEXT:       fields: Int 0, Int 0


struct S2 { int a = 10; };
constexpr S2 s2[3] = {};
// CHECK:      VarDecl {{.*}}  s2 'const S2[3]'
// CHECK-NEXT:   value: Array size=3
// CHECK-NEXT:    element: Struct
// CHECK-NEXT:      field: Int 10
// CHECK-NEXT:    element: Struct
// CHECK-NEXT:      field: Int 10
// CHECK-NEXT:    element: Struct
// CHECK-NEXT:      field: Int 10

