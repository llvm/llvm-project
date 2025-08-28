// Use --mlir-disable-threading so that the diagnostic printing is serialized.
// RUN: fir-opt %s -pass-pipeline='builtin.module(test-fir-openacc-interfaces)' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<f16 = dense<16> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, f64 = dense<64> : vector<2xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, "dlti.endianness" = "little", "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128"} {
  // This test exercises explicit-shape local array of form "arr(2:10)"
  func.func @_QPsub() {
    %c2 = arith.constant 2 : index
    %c10 = arith.constant 10 : index
    %0 = fir.alloca !fir.array<10xf32> {bindc_name = "arr", uniq_name = "_QFsubEarr"}
    %1 = fir.shape_shift %c2, %c10 : (index, index) -> !fir.shapeshift<1>
    %2 = fir.declare %0(%1) {uniq_name = "_QFsubEarr"} : (!fir.ref<!fir.array<10xf32>>, !fir.shapeshift<1>) -> !fir.ref<!fir.array<10xf32>>
    %3 = fir.embox %2(%1) : (!fir.ref<!fir.array<10xf32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<10xf32>>
    %4 = fir.box_addr %3 : (!fir.box<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>>
    %5 = acc.copyin var(%3 : !fir.box<!fir.array<10xf32>>) -> !fir.box<!fir.array<10xf32>> {name = "arr", structured = false}
    %6 = acc.copyin varPtr(%4 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "arr", structured = false}
    acc.enter_data dataOperands(%5, %6 : !fir.box<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
    return
  }

  // CHECK: Visiting: %{{.*}} = acc.copyin var(%{{.*}} : !fir.box<!fir.array<10xf32>>) -> !fir.box<!fir.array<10xf32>> {name = "arr", structured = false}
  // CHECK: Mappable: !fir.box<!fir.array<10xf32>>
  // CHECK: Type category: array
  // CHECK: Size: 40

  // CHECK: Visiting: %{{.*}} = acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "arr", structured = false}
  // CHECK: Pointer-like and Mappable: !fir.ref<!fir.array<10xf32>>
  // CHECK: Type category: array
  // CHECK: Size: 40

  // This second test exercises argument of explicit-shape arrays in following forms:
  // `real :: arr1(nn), arr2(2:nn), arr3(10)`
  // It uses the reference instead of the box in the clauses to test that bounds
  // can be generated from the shape operations.
  func.func @_QPacc_explicit_shape(%arg0: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "arr1"}, %arg1: !fir.ref<!fir.array<?xf32>> {fir.bindc_name = "arr2"}, %arg2: !fir.ref<i32> {fir.bindc_name = "nn"}) {
    %c-1 = arith.constant -1 : index
    %c2 = arith.constant 2 : index
    %c0 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %0 = fir.dummy_scope : !fir.dscope
    %1:2 = hlfir.declare %arg2 dummy_scope %0 {uniq_name = "_QFacc_explicit_shapeEnn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
    %2 = fir.alloca !fir.array<10xf32> {bindc_name = "arr3", uniq_name = "_QFacc_explicit_shapeEarr3"}
    %3 = fir.shape %c10 : (index) -> !fir.shape<1>
    %4:2 = hlfir.declare %2(%3) {uniq_name = "_QFacc_explicit_shapeEarr3"} : (!fir.ref<!fir.array<10xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<10xf32>>, !fir.ref<!fir.array<10xf32>>)
    %5 = fir.load %1#0 : !fir.ref<i32>
    %6 = fir.convert %5 : (i32) -> index
    %7 = arith.cmpi sgt, %6, %c0 : index
    %8 = arith.select %7, %6, %c0 : index
    %9 = fir.shape %8 : (index) -> !fir.shape<1>
    %10:2 = hlfir.declare %arg0(%9) dummy_scope %0 {uniq_name = "_QFacc_explicit_shapeEarr1"} : (!fir.ref<!fir.array<?xf32>>, !fir.shape<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
    %11 = arith.addi %6, %c-1 : index
    %12 = arith.cmpi sgt, %11, %c0 : index
    %13 = arith.select %12, %11, %c0 : index
    %14 = fir.shape_shift %c2, %13 : (index, index) -> !fir.shapeshift<1>
    %15:2 = hlfir.declare %arg1(%14) dummy_scope %0 {uniq_name = "_QFacc_explicit_shapeEarr2"} : (!fir.ref<!fir.array<?xf32>>, !fir.shapeshift<1>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>)
    %16 = acc.copyin var(%10#1 : !fir.ref<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>> {name = "arr1", structured = false}
    %17 = acc.copyin var(%15#1 : !fir.ref<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>> {name = "arr2", structured = false}
    %18 = acc.copyin varPtr(%4#0 : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "arr3", structured = false}
    acc.enter_data dataOperands(%16, %17, %18 : !fir.ref<!fir.array<?xf32>>, !fir.ref<!fir.array<?xf32>>, !fir.ref<!fir.array<10xf32>>)
    return
  }

  // CHECK: Visiting: %{{.*}} = acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>> {name = "arr1", structured = false}
  // CHECK: Pointer-like and Mappable: !fir.ref<!fir.array<?xf32>>
  // CHECK: Type category: array
  // CHECK: Shape: %{{.*}} = fir.shape %[[EXTENT1:.*]] : (index) -> !fir.shape<1>
  // CHECK: Bound[0]: %{{.*}} = acc.bounds lowerbound(%[[LB1:.*]] : index) upperbound(%[[UB1:.*]] : index) extent(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%c1{{.*}} : index)
  // CHECK: Lower bound: %[[LB1]] = arith.constant 0 : index
  // CHECK: Upper bound: %[[UB1]] = arith.subi %[[EXTENT1]], %c1{{.*}} : index

  // CHECK: Visiting: %{{.*}} = acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>> {name = "arr2", structured = false}
  // CHECK: Pointer-like and Mappable: !fir.ref<!fir.array<?xf32>>
  // CHECK: Type category: array
  // CHECK: Shape: %{{.*}} = fir.shape_shift %c2{{.*}}, %[[EXTENT2:.*]] : (index, index) -> !fir.shapeshift<1>
  // CHECK: Bound[0]: %{{.*}} = acc.bounds lowerbound(%[[LB2:.*]] : index) upperbound(%[[UB2:.*]] : index) extent(%{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%c2{{.*}} : index)
  // CHECK: Lower bound: %[[LB2]] = arith.constant 0 : index
  // CHECK: Upper bound: %[[UB2]] = arith.subi %[[EXTENT2]], %c1{{.*}} : index

  // CHECK: Visiting: %{{.*}} = acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.array<10xf32>>) -> !fir.ref<!fir.array<10xf32>> {name = "arr3", structured = false}
  // CHECK: Pointer-like and Mappable: !fir.ref<!fir.array<10xf32>>
  // CHECK: Type category: array
  // CHECK: Size: 40
  // CHECK: Offset: 0
  // CHECK: Shape: %{{.*}} = fir.shape %[[EXTENT3:.*]] : (index) -> !fir.shape<1>
  // CHECK: Bound[0]: %{{.*}} = acc.bounds lowerbound(%[[LB3:.*]] : index) upperbound(%[[UB3:.*]] : index) extent(%c10{{.*}} : index) stride(%c1{{.*}} : index) startIdx(%c1{{.*}} : index)
  // CHECK: Lower bound: %[[LB3]] = arith.constant 0 : index
  // CHECK: Upper bound: %[[UB3]] = arith.subi %[[EXTENT3]], %c1{{.*}} : index
}
