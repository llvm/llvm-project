// RUN: fir-opt --mif-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 23.0.0 (git@github.com:SiPearl/llvm-project.git d31a4730513391710d91c5ad33bb8ea3d68db3cb)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
// CHECK-LABEL: func.func @_QQmain
 func.func @_QQmain() attributes {fir.bindc_name = "TEST"} {
    %0 = fir.alloca !fir.array<2xi64>
    %1 = fir.alloca !fir.array<3xi64>
    %2 = fir.dummy_scope : !fir.dscope
    %3 = fir.address_of(@_QFEa) : !fir.ref<!fir.box<!fir.heap<i32>, corank:3>>
    %4:2 = hlfir.declare %3 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<i32>, corank:3>>) -> (!fir.ref<!fir.box<!fir.heap<i32>, corank:3>>, !fir.ref<!fir.box<!fir.heap<i32>, corank:3>>)
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
    mif.alloc_coarray %4#0 lcobounds %19 ucobounds %20 errmsg %10 {uniq_name = "_QFEa"} : (!fir.ref<!fir.box<!fir.heap<i32>, corank:3>>, !fir.box<!fir.array<3xi64>>, !fir.box<!fir.array<2xi64>>, !fir.box<none>) -> ()
    %21 = fir.load %4#0 : !fir.ref<!fir.box<!fir.heap<i32>, corank:3>>
    %22 = fir.alloca !fir.array<3xi32>
    %c1_i32 = arith.constant 1 : i32
    %23 = mif.lcobound coarray %21 dim %c1_i32 : (!fir.box<!fir.heap<i32>, corank:3>, i32) -> i32
    %c0_0 = arith.constant 0 : index
    %24 = fir.coordinate_of %22, %c0_0 : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
    fir.store %23 to %24 : !fir.ref<i32>
    %c2_i32_1 = arith.constant 2 : i32
    %25 = mif.lcobound coarray %21 dim %c2_i32_1 : (!fir.box<!fir.heap<i32>, corank:3>, i32) -> i32
    %c1_2 = arith.constant 1 : index
    %26 = fir.coordinate_of %22, %c1_2 : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
    fir.store %25 to %26 : !fir.ref<i32>
    %c3_i32_3 = arith.constant 3 : i32
    %27 = mif.lcobound coarray %21 dim %c3_i32_3 : (!fir.box<!fir.heap<i32>, corank:3>, i32) -> i32
    %c2_4 = arith.constant 2 : index
    %28 = fir.coordinate_of %22, %c2_4 : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
    fir.store %27 to %28 : !fir.ref<i32>
    %29 = fir.embox %22 : (!fir.ref<!fir.array<3xi32>>) -> !fir.box<!fir.array<3xi32>>
    %30:2 = hlfir.declare %29 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.array<3xi32>>) -> (!fir.box<!fir.array<3xi32>>, !fir.box<!fir.array<3xi32>>)
    %false = arith.constant false
    %31 = hlfir.as_expr %30#0 move %false : (!fir.box<!fir.array<3xi32>>, i1) -> !hlfir.expr<3xi32>
    hlfir.assign %31 to %7#0 : !hlfir.expr<3xi32>, !fir.ref<!fir.array<3xi32>>
    hlfir.destroy %31 : !hlfir.expr<3xi32>
    %c2_i32_5 = arith.constant 2 : i32
    %32 = fir.load %4#0 : !fir.ref<!fir.box<!fir.heap<i32>, corank:3>>
    %33 = mif.lcobound coarray %32 dim %c2_i32_5 : (!fir.box<!fir.heap<i32>, corank:3>, i32) -> i32
    hlfir.assign %33 to %9#0 : i32, !fir.ref<i32>
    %34 = fir.load %4#0 : !fir.ref<!fir.box<!fir.heap<i32>, corank:3>>
    %35 = fir.alloca !fir.array<3xi32>
    %c1_i32_6 = arith.constant 1 : i32
    %36 = mif.ucobound coarray %34 dim %c1_i32_6 : (!fir.box<!fir.heap<i32>, corank:3>, i32) -> i32
    %c0_7 = arith.constant 0 : index
    %37 = fir.coordinate_of %35, %c0_7 : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
    fir.store %36 to %37 : !fir.ref<i32>
    %c2_i32_8 = arith.constant 2 : i32
    %38 = mif.ucobound coarray %34 dim %c2_i32_8 : (!fir.box<!fir.heap<i32>, corank:3>, i32) -> i32
    %c1_9 = arith.constant 1 : index
    %39 = fir.coordinate_of %35, %c1_9 : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
    fir.store %38 to %39 : !fir.ref<i32>
    %c3_i32_10 = arith.constant 3 : i32
    %40 = mif.ucobound coarray %34 dim %c3_i32_10 : (!fir.box<!fir.heap<i32>, corank:3>, i32) -> i32
    %c2_11 = arith.constant 2 : index
    %41 = fir.coordinate_of %35, %c2_11 : (!fir.ref<!fir.array<3xi32>>, index) -> !fir.ref<i32>
    fir.store %40 to %41 : !fir.ref<i32>
    %42 = fir.embox %35 : (!fir.ref<!fir.array<3xi32>>) -> !fir.box<!fir.array<3xi32>>
    %43:2 = hlfir.declare %42 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.array<3xi32>>) -> (!fir.box<!fir.array<3xi32>>, !fir.box<!fir.array<3xi32>>)
    %false_12 = arith.constant false
    %44 = hlfir.as_expr %43#0 move %false_12 : (!fir.box<!fir.array<3xi32>>, i1) -> !hlfir.expr<3xi32>
    hlfir.assign %44 to %7#0 : !hlfir.expr<3xi32>, !fir.ref<!fir.array<3xi32>>
    hlfir.destroy %44 : !hlfir.expr<3xi32>
    %c2_i32_13 = arith.constant 2 : i32
    %45 = fir.load %4#0 : !fir.ref<!fir.box<!fir.heap<i32>, corank:3>>
    %46 = mif.ucobound coarray %45 dim %c2_i32_13 : (!fir.box<!fir.heap<i32>, corank:3>, i32) -> i32
    hlfir.assign %46 to %9#0 : i32, !fir.ref<i32>
    return
  }
}

// CHECK: fir.call @_QMprifPprif_lcobound_with_dim
// CHECK: fir.call @_QMprifPprif_lcobound_with_dim
// CHECK: fir.call @_QMprifPprif_lcobound_with_dim
// CHECK: fir.call @_QMprifPprif_ucobound_with_dim
// CHECK: fir.call @_QMprifPprif_ucobound_with_dim
// CHECK: fir.call @_QMprifPprif_ucobound_with_dim
