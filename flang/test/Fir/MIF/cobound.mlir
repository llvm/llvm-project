// RUN: fir-opt --mif-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 23.0.0 (git@github.com:SiPearl/llvm-project.git d31a4730513391710d91c5ad33bb8ea3d68db3cb)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
// CHECK-LABEL: func.func @_QQmain
 func.func @_QQmain() attributes {fir.bindc_name = "TEST"} {
    %0 = fir.alloca !fir.array<2xi64>
    %1 = fir.alloca !fir.array<3xi64>
    %2 = fir.dummy_scope : !fir.dscope
    %3 = fir.address_of(@_QFEa) : !fir.ref<!fir.box<!fir.heap<i32>>>
    %4:2 = hlfir.declare %3 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
    %c3 = arith.constant 3 : index
    %5 = fir.alloca !fir.array<3xi32> {bindc_name = "res1", uniq_name = "_QFEres1"}
    %6 = fir.shape %c3 : (index) -> !fir.shape<1>
    %7:2 = hlfir.declare %5(%6) {uniq_name = "_QFEres1"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
    %8 = fir.alloca i32 {bindc_name = "res2", uniq_name = "_QFEres2"}
    %9:2 = hlfir.declare %8 {uniq_name = "_QFEres2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %10 = fir.absent !fir.box<none>
    %c1_i64 = arith.constant 1 : i64
    %c2_i32 = arith.constant 2 : i32
    %11 = fir.convert %c2_i32 : (i32) -> i64
    %c0 = arith.constant 0 : index
    %12 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<3xi64>>, index) -> !fir.ref<i64>
    fir.store %c1_i64 to %12 : !fir.ref<i64>
    %13 = fir.coordinate_of %0, %c0 : (!fir.ref<!fir.array<2xi64>>, index) -> !fir.ref<i64>
    fir.store %11 to %13 : !fir.ref<i64>
    %c3_i32 = arith.constant 3 : i32
    %14 = fir.convert %c3_i32 : (i32) -> i64
    %c5_i32 = arith.constant 5 : i32
    %15 = fir.convert %c5_i32 : (i32) -> i64
    %c1 = arith.constant 1 : index
    %16 = fir.coordinate_of %1, %c1 : (!fir.ref<!fir.array<3xi64>>, index) -> !fir.ref<i64>
    fir.store %14 to %16 : !fir.ref<i64>
    %17 = fir.coordinate_of %0, %c1 : (!fir.ref<!fir.array<2xi64>>, index) -> !fir.ref<i64>
    fir.store %15 to %17 : !fir.ref<i64>
    %c2 = arith.constant 2 : index
    %18 = fir.coordinate_of %1, %c2 : (!fir.ref<!fir.array<3xi64>>, index) -> !fir.ref<i64>
    fir.store %c1_i64 to %18 : !fir.ref<i64>
    %19 = fir.embox %1 : (!fir.ref<!fir.array<3xi64>>) -> !fir.box<!fir.array<3xi64>>
    %20 = fir.embox %0 : (!fir.ref<!fir.array<2xi64>>) -> !fir.box<!fir.array<2xi64>>
    mif.alloc_coarray %4#0 lcobounds %19 ucobounds %20 errmsg %10 {uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.box<!fir.array<3xi64>>, !fir.box<!fir.array<2xi64>>, !fir.box<none>) -> ()
    %21 = fir.load %4#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
    %22 = fir.box_addr %21 {fir.corank = 3 : i32} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
// CHECK: fir.call @_QMprifPprif_lcobound_no_dim
    %23 = mif.lcobound coarray %22 : (!fir.heap<i32>) -> !fir.box<!fir.array<?xi64>>
    %24:2 = hlfir.declare %23 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.array<?xi64>>) -> (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>)
    %false = arith.constant false
    %25 = hlfir.as_expr %24#0 move %false : (!fir.box<!fir.array<?xi64>>, i1) -> !hlfir.expr<?xi64>
    %c0_0 = arith.constant 0 : index
    %26:3 = fir.box_dims %24#0, %c0_0 : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
    %27 = fir.shape %26#1 : (index) -> !fir.shape<1>
    %28 = hlfir.elemental %27 unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
    ^bb0(%arg0: index):
      %43 = hlfir.apply %25, %arg0 : (!hlfir.expr<?xi64>, index) -> i64
      %44 = fir.convert %43 : (i64) -> i32
      hlfir.yield_element %44 : i32
    }
    hlfir.assign %28 to %7#0 : !hlfir.expr<?xi32>, !fir.ref<!fir.array<3xi32>>
    hlfir.destroy %28 : !hlfir.expr<?xi32>
    hlfir.destroy %25 : !hlfir.expr<?xi64>
    %c2_i32_1 = arith.constant 2 : i32
    %29 = fir.load %4#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
    %30 = fir.box_addr %29 {fir.corank = 3 : i32} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
// CHECK: fir.call @_QMprifPprif_lcobound_with_dim
    %31 = mif.lcobound coarray %30 dim %c2_i32_1 : (!fir.heap<i32>, i32) -> i32
    hlfir.assign %31 to %9#0 : i32, !fir.ref<i32>
    %32 = fir.load %4#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
    %33 = fir.box_addr %32 {fir.corank = 3 : i32} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
    %34 = mif.ucobound coarray %33 : (!fir.heap<i32>) -> !fir.box<!fir.array<?xi64>>
// CHECK: fir.call @_QMprifPprif_ucobound_no_dim
    %35:2 = hlfir.declare %34 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.array<?xi64>>) -> (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>)
    %false_2 = arith.constant false
    %36 = hlfir.as_expr %35#0 move %false_2 : (!fir.box<!fir.array<?xi64>>, i1) -> !hlfir.expr<?xi64>
    %c0_3 = arith.constant 0 : index
    %37:3 = fir.box_dims %35#0, %c0_3 : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
    %38 = fir.shape %37#1 : (index) -> !fir.shape<1>
    %39 = hlfir.elemental %38 unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
    ^bb0(%arg0: index):
      %43 = hlfir.apply %36, %arg0 : (!hlfir.expr<?xi64>, index) -> i64
      %44 = fir.convert %43 : (i64) -> i32
      hlfir.yield_element %44 : i32
    }
    hlfir.assign %39 to %7#0 : !hlfir.expr<?xi32>, !fir.ref<!fir.array<3xi32>>
    hlfir.destroy %39 : !hlfir.expr<?xi32>
    hlfir.destroy %36 : !hlfir.expr<?xi64>
    %c2_i32_4 = arith.constant 2 : i32
    %40 = fir.load %4#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
    %41 = fir.box_addr %40 {fir.corank = 3 : i32} : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
// CHECK: fir.call @_QMprifPprif_ucobound_with_dim
    %42 = mif.ucobound coarray %41 dim %c2_i32_4 : (!fir.heap<i32>, i32) -> i32
    hlfir.assign %42 to %9#0 : i32, !fir.ref<i32>
    return
  }
}
