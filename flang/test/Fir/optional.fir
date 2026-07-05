// RUN: tco %s | FileCheck %s
// RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck %s

// Test fir.is_present and fir.absent codegen

// CHECK-LABEL: @foo1
func.func @foo1(%arg0: !fir.box<!fir.array<?xf32>>) -> i1 {
  // CHECK: %[[ptr:.*]] = ptrtoint ptr %{{.*}} to i64
  // CHECK: icmp ne i64 %[[ptr]], 0
  %0 = fir.is_present %arg0 : (!fir.box<!fir.array<?xf32>>) -> i1
  return %0 : i1
}

// CHECK-LABEL: @bar1
func.func @bar1() -> i1 {
  %0 = fir.absent !fir.box<!fir.array<?xf32>>
  // CHECK: call i1 @foo1(ptr null)
  %1 = fir.call @foo1(%0) : (!fir.box<!fir.array<?xf32>>) -> i1
  return %1 : i1
}

// CHECK-LABEL: @foo2
func.func @foo2(%arg0: !fir.ref<i64>) -> i1 {
  // CHECK: %[[ptr:.*]] = ptrtoint ptr %{{.*}} to i64
  // CHECK: icmp ne i64 %[[ptr]], 0
  %0 = fir.is_present %arg0 : (!fir.ref<i64>) -> i1
  return %0 : i1
}

// CHECK-LABEL: @bar2
func.func @bar2() -> i1 {
  %0 = fir.absent !fir.ref<i64>
  // CHECK: call i1 @foo2(ptr null)
  %1 = fir.call @foo2(%0) : (!fir.ref<i64>) -> i1
  return %1 : i1
}

// CHECK-LABEL: @foo3
func.func @foo3(%arg0: !fir.boxchar<1>) -> i1 {
  // CHECK: %[[extract:.*]] = extractvalue { ptr, i64 } %{{.*}}, 0
  // CHECK: %[[ptr:.*]] = ptrtoint ptr %[[extract]] to i64
  // CHECK: icmp ne i64 %[[ptr]], 0
  %0 = fir.is_present %arg0 : (!fir.boxchar<1>) -> i1
  return %0 : i1
}

// CHECK-LABEL: @bar3
func.func @bar3() -> i1 {
  %0 = fir.absent !fir.boxchar<1>
  // CHECK: call i1 @foo3(ptr null, i64 0)
  %1 = fir.call @foo3(%0) : (!fir.boxchar<1>) -> i1
  return %1 : i1
}

// CHECK-LABEL: @foo4(
// CHECK-SAME: ptr %[[arg:.*]])
func.func @foo4(%arg0: !fir.boxproc<(i32)->(i64)>) -> i1 {
  // CHECK: %[[ptr:.*]] = ptrtoint ptr %[[arg]] to i64
  // CHECK: icmp ne i64 %[[ptr]], 0
  %0 = fir.is_present %arg0 : (!fir.boxproc<(i32)->(i64)>) -> i1
  return %0 : i1
}

// CHECK-LABEL: @bar4
func.func @bar4() -> i1 {
  %0 = fir.absent !fir.boxproc<(i32)->(i64)>
  // CHECK: call i1 @foo4(ptr null)
  %1 = fir.call @foo4(%0) : (!fir.boxproc<(i32)->(i64)>) -> i1
  return %1 : i1
}

// CHECK-LABEL: @foo5(
// CHECK-SAME: ptr %[[arg:.*]])
func.func @foo5(%arg0: (i32)->(i64)) -> i1 {
  // CHECK: %[[ptr:.*]] = ptrtoint ptr %[[arg]] to i64
  // CHECK: icmp ne i64 %[[ptr]], 0
  %0 = fir.is_present %arg0 : ((i32)->(i64)) -> i1
  return %0 : i1
}

// CHECK-LABEL: @bar5
func.func @bar5() -> i1 {
  %0 = fir.absent (i32)->(i64)
  // CHECK: call i1 @foo5(ptr null)
  %1 = fir.call @foo5(%0) : ((i32)->(i64)) -> i1
  return %1 : i1
}
