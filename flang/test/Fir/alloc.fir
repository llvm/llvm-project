// RUN: tco -emit-fir %s | tco | FileCheck %s

// CHECK-LABEL: define i32* @f1()
func @f1() -> !fir.ref<i32> {
  // CHECK: alloca i32, i64 1
  %1 = fir.alloca i32
  return %1 : !fir.ref<i32>
}

// CHECK-LABEL: define i32* @f2()
func @f2() -> !fir.ref<i32> {
  %0 = constant 100 : index
  // CHECK: alloca i32, i64 100
  %1 = fir.alloca i32, %0
  return %1 : !fir.ref<i32>
}

// CHECK-LABEL: define i32* @f3()
func @f3() -> !fir.heap<i32> {
  // CHECK: call i8* @malloc(i64 4)
  %1 = fir.allocmem i32
  return %1 : !fir.heap<i32>
}

// CHECK-LABEL: define i32* @f4()
func @f4() -> !fir.heap<i32> {
  %0 = constant 100 : index
  // CHECK: call i8* @malloc(i64 400)
  %1 = fir.allocmem i32, %0
  return %1 : !fir.heap<i32>
}
