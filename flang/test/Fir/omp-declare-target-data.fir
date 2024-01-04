// RUN: fir-opt --fir-to-llvm-ir %s | FileCheck %s

module attributes {omp.is_target_device = false} {

  // CHECK: llvm.mlir.global external @_QMtest_0Earray_1d(dense<[1, 2, 3]> : tensor<3xi32>) {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>{{.*}}} : !llvm.array<3 x i32>
  fir.global @_QMtest_0Earray_1d(dense<[1, 2, 3]> : tensor<3xi32>) {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : !fir.array<3xi32>

  // CHECK: llvm.mlir.global external @_QMtest_0Earray_2d() {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>{{.*}}} : !llvm.array<2 x array<2 x i32>>
  fir.global @_QMtest_0Earray_2d {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : !fir.array<2x2xi32> {
    %0 = fir.undefined !fir.array<2x2xi32>
    %c1_i32 = arith.constant 1 : i32
    %1 = fir.insert_value %0, %c1_i32, [0 : index, 0 : index] : (!fir.array<2x2xi32>, i32) -> !fir.array<2x2xi32>
    %c2_i32 = arith.constant 2 : i32
    %2 = fir.insert_value %1, %c2_i32, [1 : index, 0 : index] : (!fir.array<2x2xi32>, i32) -> !fir.array<2x2xi32>
    %c3_i32 = arith.constant 3 : i32
    %3 = fir.insert_value %2, %c3_i32, [0 : index, 1 : index] : (!fir.array<2x2xi32>, i32) -> !fir.array<2x2xi32>
    %c4_i32 = arith.constant 4 : i32
    %4 = fir.insert_value %3, %c4_i32, [1 : index, 1 : index] : (!fir.array<2x2xi32>, i32) -> !fir.array<2x2xi32>
    %c2 = arith.constant 2 : index
    %c2_0 = arith.constant 2 : index
    fir.has_value %4 : !fir.array<2x2xi32>
  }

  // CHECK: llvm.mlir.global external @_QMtest_0Edata_extended_link_1() {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>{{.*}}} : f32
  fir.global @_QMtest_0Edata_extended_link_1 {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : f32 {
    %cst = arith.constant 2.000000e+00 : f32
    fir.has_value %cst : f32
  }

  // CHECK: llvm.mlir.global external @_QMtest_0Edata_extended_link_2() {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>{{.*}}} : f32
  fir.global @_QMtest_0Edata_extended_link_2 {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : f32 {
    %cst = arith.constant 3.000000e+00 : f32
    fir.has_value %cst : f32
  }

  // CHECK: llvm.mlir.global external @_QMtest_0Edata_extended_to_1() {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>{{.*}}} : f32
  fir.global @_QMtest_0Edata_extended_to_1 {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : f32 {
    %cst = arith.constant 2.000000e+00 : f32
    fir.has_value %cst : f32
  }

  // CHECK: llvm.mlir.global external @_QMtest_0Edata_extended_to_2() {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>{{.*}}} : f32 {
  fir.global @_QMtest_0Edata_extended_to_2 {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : f32 {
    %cst = arith.constant 3.000000e+00 : f32
    fir.has_value %cst : f32
  }

  // CHECK: llvm.mlir.global external @_QMtest_0Edata_int() {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>{{.*}}} : i32
  fir.global @_QMtest_0Edata_int {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : i32 {
    %c10_i32 = arith.constant 10 : i32
    fir.has_value %c10_i32 : i32
  }

  // CHECK: llvm.mlir.global external @_QMtest_0Edata_int_clauseless() {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>{{.*}}} : i32
  fir.global @_QMtest_0Edata_int_clauseless {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : i32 {
    %c1_i32 = arith.constant 1 : i32
    fir.has_value %c1_i32 : i32
  }

  // CHECK: llvm.mlir.global external @_QMtest_0Edata_int_to() {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>{{.*}}} : i32
  fir.global @_QMtest_0Edata_int_to {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (to)>} : i32 {
    %c5_i32 = arith.constant 5 : i32
    fir.has_value %c5_i32 : i32
  }

  // CHECK: llvm.mlir.global external @_QMtest_0Ept1() {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>{{.*}}} : !llvm.struct<(ptr, i64, i32, i8, i8, i8, i8)> {
  fir.global @_QMtest_0Ept1 {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} : !fir.box<!fir.ptr<i32>> {
    %0 = fir.zero_bits !fir.ptr<i32>
    %1 = fir.embox %0 : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
    fir.has_value %1 : !fir.box<!fir.ptr<i32>>
  }

  // CHECK: llvm.mlir.global external @_QMtest_0Ept2_tar() {{{.*}}omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>{{.*}}} : i32
  fir.global @_QMtest_0Ept2_tar {omp.declare_target = #omp.declaretarget<device_type = (any), capture_clause = (link)>} target : i32 {
    %c5_i32 = arith.constant 5 : i32
    fir.has_value %c5_i32 : i32
  }
}
