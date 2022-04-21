// RUN: tco %s | FileCheck %s

// Test fir.is_present and fir.absent codegen

// CHECK-LABEL: @foo1
func.func @foo1(%arg0: !fir.box<!fir.array<?xf32>>) -> i1 {
  // CHECK: %[[ptr:.*]] = ptrtoint { float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* %{{.*}} to i64
  // CHECK: icmp ne i64 %[[ptr]], 0
  %0 = fir.is_present %arg0 : (!fir.box<!fir.array<?xf32>>) -> i1
  return %0 : i1
}

// CHECK-LABEL: @bar1
func.func @bar1() -> i1 {
  %0 = fir.absent !fir.box<!fir.array<?xf32>>
  // CHECK: call i1 @foo1({ float*, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }* null)
  %1 = fir.call @foo1(%0) : (!fir.box<!fir.array<?xf32>>) -> i1
  return %1 : i1
}

// CHECK-LABEL: @foo2
func.func @foo2(%arg0: !fir.ref<i64>) -> i1 {
  // CHECK: %[[ptr:.*]] = ptrtoint i64* %{{.*}} to i64
  // CHECK: icmp ne i64 %[[ptr]], 0
  %0 = fir.is_present %arg0 : (!fir.ref<i64>) -> i1
  return %0 : i1
}

// CHECK-LABEL: @bar2
func.func @bar2() -> i1 {
  %0 = fir.absent !fir.ref<i64>
  // CHECK: call i1 @foo2(i64* null)
  %1 = fir.call @foo2(%0) : (!fir.ref<i64>) -> i1
  return %1 : i1
}

// CHECK-LABEL: @foo3
func.func @foo3(%arg0: !fir.boxchar<1>) -> i1 {
  // CHECK: %[[extract:.*]] = extractvalue { i8*, i64 } %{{.*}}, 0
  // CHECK: %[[ptr:.*]] = ptrtoint i8* %[[extract]] to i64
  // CHECK: icmp ne i64 %[[ptr]], 0
  %0 = fir.is_present %arg0 : (!fir.boxchar<1>) -> i1
  return %0 : i1
}

// CHECK-LABEL: @bar3
func.func @bar3() -> i1 {
  %0 = fir.absent !fir.boxchar<1>
  // CHECK: call i1 @foo3(i8* null, i64 undef)
  %1 = fir.call @foo3(%0) : (!fir.boxchar<1>) -> i1
  return %1 : i1
}
