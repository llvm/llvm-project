// RUN: %clang_cc1 -std=c23 -ast-print %s | FileCheck %s
// RUN: %clang_cc1 -std=c23 -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c23 -include-pch %t -ast-print -x c /dev/null | FileCheck %s

// CHECK-LABEL: int f1(void)
// CHECK: return (constexpr int){1} + (static _Thread_local int){2} + (register int){3} + (constexpr const int){4};
int f1(void) {
  return (constexpr int){1} + (_Thread_local static int){2} + (register int){3} +
         (constexpr const int){4};
}

// CHECK-LABEL: int f2(void)
// CHECK: return (int[3]){1, 2, 3}[0] + (constexpr int[3]){1, 2, 3}[0];
int f2(void) {
  return (int[3]){1, 2, 3}[0] + (constexpr int[3]){1, 2, 3}[0];
}

typedef const int T;

// CHECK-LABEL: int f3(void)
// CHECK: return (constexpr T){1} + (constexpr int[]){1}[0] + (constexpr const int[]){1}[0] + (constexpr T[]){1}[0];
int f3(void) {
  return (constexpr T){1} + (constexpr int[]){1}[0] +
         (constexpr const int[]){1}[0] + (constexpr T[]){1}[0];
}

// CHECK-LABEL: void f4(void)
// CHECK: typedef typeof (*(int (*)[(static _Thread_local int){1}])0) T1;
// CHECK: typedef typeof (*(int (*)[(register int){1}])0) T2;
void f4(void) {
  typedef typeof(*(int (*)[(static thread_local int){1}])0) T1;
  typedef typeof(*(int (*)[(register int){1}])0) T2;
}
