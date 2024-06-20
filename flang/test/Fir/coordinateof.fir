// RUN: fir-opt %s | tco | FileCheck %s

// tests on coordinate_of op

// CHECK-LABEL: @foo1
func.func @foo1(%i : i32, %j : i32, %k : i32) -> !fir.ref<f32> {
  %1 = fir.alloca !fir.array<10 x 20 x 30 x f32>
  // CHECK: %[[alloca:.*]] = alloca [30 x [20 x [10 x float]]]
  %2 = fir.convert %1 : (!fir.ref<!fir.array<10 x 20 x 30 x f32>>) -> !fir.ref<!fir.array<10 x 20 x ? x f32>>
  // CHECK: getelementptr [20 x [10 x float]], ptr %[[alloca]]
  %3 = fir.coordinate_of %2, %i, %j, %k : (!fir.ref<!fir.array<10 x 20 x ? x f32>>, i32, i32, i32) -> !fir.ref<f32>
  return %3 : !fir.ref<f32>
}

// CHECK-LABEL: @foo2
func.func @foo2(%i : i32, %j : i32, %k : i32) -> !fir.ref<f32> {
  %1 = fir.alloca !fir.array<10 x 20 x 30 x f32>
  // CHECK: %[[alloca:.*]] = alloca [30 x [20 x [10 x float]]]
  %2 = fir.convert %1 : (!fir.ref<!fir.array<10 x 20 x 30 x f32>>) -> !fir.ref<!fir.array<?xf32>>
  // CHECK: getelementptr float, ptr %[[alloca]]
  %3 = fir.coordinate_of %2, %i : (!fir.ref<!fir.array<?xf32>>, i32) -> !fir.ref<f32>
  return %3 : !fir.ref<f32>
}

// CHECK-LABEL: @foo3
func.func @foo3(%box : !fir.box<!fir.array<?xi32>>, %i : i32) -> i32 {
  // CHECK: %[[cvt:.*]] = sext i32 %
  %ii = fir.convert %i : (i32) -> index
  // CHECK: %[[gep0:.*]] = getelementptr { ptr
  // CHECK: %[[boxptr:.*]] = load ptr, ptr %[[gep0]]
  // CHECK: %[[gep1:.*]] = getelementptr { ptr, i64, {{.*}} i32 7
  // CHECK: %[[stride:.*]] = load i64, ptr %[[gep1]]
  // CHECK: %[[dimoffset:.*]] = mul nsw i64 %[[cvt]], %[[stride]]
  // CHECK: %[[offset:.*]] = add nsw i64 %[[dimoffset]], 0
  // CHECK: %[[gep2:.*]] = getelementptr i8, ptr %[[boxptr]], i64 %[[offset]]
  %1 = fir.coordinate_of %box, %ii : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
  // CHECK: load i32, ptr %[[gep2]]
  %rv = fir.load %1 : !fir.ref<i32>
  return %rv : i32
}

// CHECK-LABEL: @foo4
func.func @foo4(%a : !fir.ptr<!fir.array<5x15x25xi32>>, %i : i32, %j : i64, %k : index) -> i32 {
  // CHECK: getelementptr [25 x [15 x [5 x
  %1 = fir.coordinate_of %a, %k : (!fir.ptr<!fir.array<5x15x25xi32>>, index) -> !fir.ref<!fir.array<5x15xi32>>
  // CHECK: getelementptr [15 x [5 x
  %2 = fir.coordinate_of %1, %j : (!fir.ref<!fir.array<5x15xi32>>, i64) -> !fir.ref<!fir.array<5xi32>>
  // CHECK: %[[ref:.*]] = getelementptr [5 x
  %3 = fir.coordinate_of %2, %i : (!fir.ref<!fir.array<5xi32>>, i32) -> !fir.ref<i32>
  // CHECK: load i32, ptr %[[ref]]
  %4 = fir.load %3 : !fir.ref<i32>
  return %4 : i32
}

// CHECK-LABEL: @foo5
func.func @foo5(%box : !fir.box<!fir.ptr<!fir.array<?xi32>>>, %i : index) -> i32 {
  // similar to foo3 test. Just check that the ptr type is not disturbing codegen.
  %1 = fir.coordinate_of %box, %i : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> !fir.ref<i32>
  // CHECK: load i32, ptr %{{.*}}
  %rv = fir.load %1 : !fir.ref<i32>
  return %rv : i32
}

// CHECK-LABEL: @foo6
// CHECK-SAME: (ptr %[[box:.*]], i64 %{{.*}}, ptr %{{.*}}) 
func.func @foo6(%box : !fir.box<!fir.ptr<!fir.array<?x!fir.char<1>>>>, %i : i64 , %res : !fir.ref<!fir.char<1>>) {
  // CHECK: %[[addr_gep:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[box]], i32 0, i32 0
  // CHECK: %[[addr:.*]] = load ptr, ptr %[[addr_gep]]
  // CHECK: %[[stride_gep:.*]] = getelementptr { ptr, i64, i32, i8, i8, i8, i8, [1 x [3 x i64]] }, ptr %[[box]], i32 0, i32 7, i32 0, i32 2
  // CHECK: %[[stride:.*]] = load i64, ptr %[[stride_gep]]
  // CHECK: %[[mul:.*]] = mul nsw i64 %{{.*}}, %[[stride]]
  // CHECK: %[[offset:.*]] = add nsw i64 %[[mul]], 0
  // CHECK: %[[gep:.*]] = getelementptr i8, ptr %[[addr]], i64 %[[offset]]
  %coor = fir.coordinate_of %box, %i : (!fir.box<!fir.ptr<!fir.array<?x!fir.char<1>>>>, i64) -> !fir.ref<!fir.char<1>>

  // CHECK: load [1 x i8], ptr %[[gep]]
  %load = fir.load %coor : !fir.ref<!fir.char<1>>
  fir.store %load to %res : !fir.ref<!fir.char<1>>
  return
}
