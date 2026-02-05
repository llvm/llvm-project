// RUN: fir-opt --mif-convert %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 8, 16, 32, 64>, "dlti.stack_alignment" = 128 : i64>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang version 23.0.0 (git@github.com:SiPearl/llvm-project.git d31a4730513391710d91c5ad33bb8ea3d68db3cb)", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func @_QQmain() attributes {fir.bindc_name = "TEST"} {
    %0 = fir.alloca !fir.array<2xi64>
    %1 = fir.alloca !fir.array<3xi64>
    %2 = fir.dummy_scope : !fir.dscope
    %3 = fir.address_of(@_QFEa) : !fir.ref<i32>
    %c1_i64 = arith.constant 1 : i64
    %c0 = arith.constant 0 : index
    %4 = fir.coordinate_of %1, %c0 : (!fir.ref<!fir.array<3xi64>>, index) -> !fir.ref<i64>
    fir.store %c1_i64 to %4 : !fir.ref<i64>
    %c3_i64 = arith.constant 3 : i64
    %c1 = arith.constant 1 : index
    %5 = fir.coordinate_of %1, %c1 : (!fir.ref<!fir.array<3xi64>>, index) -> !fir.ref<i64>
    fir.store %c3_i64 to %5 : !fir.ref<i64>
    %c1_i64_0 = arith.constant 1 : i64
    %c2 = arith.constant 2 : index
    %6 = fir.coordinate_of %1, %c2 : (!fir.ref<!fir.array<3xi64>>, index) -> !fir.ref<i64>
    fir.store %c1_i64_0 to %6 : !fir.ref<i64>
    %7 = fir.embox %1 : (!fir.ref<!fir.array<3xi64>>) -> !fir.box<!fir.array<3xi64>>
    %c1_i64_1 = arith.constant 1 : i64
    %c0_2 = arith.constant 0 : index
    %8 = fir.coordinate_of %0, %c0_2 : (!fir.ref<!fir.array<2xi64>>, index) -> !fir.ref<i64>
    fir.store %c1_i64_1 to %8 : !fir.ref<i64>
    %c3_i64_3 = arith.constant 3 : i64
    %c1_4 = arith.constant 1 : index
    %9 = fir.coordinate_of %0, %c1_4 : (!fir.ref<!fir.array<2xi64>>, index) -> !fir.ref<i64>
    fir.store %c3_i64_3 to %9 : !fir.ref<i64>
    %10 = fir.embox %0 : (!fir.ref<!fir.array<2xi64>>) -> !fir.box<!fir.array<2xi64>>
    mif.alloc_coarray %3 lcobounds %7 ucobounds %10 {uniq_name = "_QFEa"} : (!fir.ref<i32>, !fir.box<!fir.array<3xi64>>, !fir.box<!fir.array<2xi64>>) -> ()
    %11:2 = hlfir.declare %3 {fir.corank = 3 : i32, uniq_name = "_QFEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %c3 = arith.constant 3 : index
    %12 = fir.alloca !fir.array<3xi32> {bindc_name = "res", uniq_name = "_QFEres"}
    %13 = fir.shape %c3 : (index) -> !fir.shape<1>
    %14:2 = hlfir.declare %12(%13) {uniq_name = "_QFEres"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
    %c3_5 = arith.constant 3 : index
    %15 = fir.alloca !fir.array<3xi64> {bindc_name = "res2", uniq_name = "_QFEres2"}
    %16 = fir.shape %c3_5 : (index) -> !fir.shape<1>
    %17:2 = hlfir.declare %15(%16) {uniq_name = "_QFEres2"} : (!fir.ref<!fir.array<3xi64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi64>>, !fir.ref<!fir.array<3xi64>>)
    %18 = mif.coshape coarray %11#0 : (!fir.ref<i32>) -> !fir.box<!fir.array<?xi64>>
    %19:2 = hlfir.declare %18 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.array<?xi64>>) -> (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>)
    %false = arith.constant false
    %20 = hlfir.as_expr %19#0 move %false : (!fir.box<!fir.array<?xi64>>, i1) -> !hlfir.expr<?xi64>
    %c0_6 = arith.constant 0 : index
    %21:3 = fir.box_dims %19#0, %c0_6 : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
    %22 = fir.shape %21#1 : (index) -> !fir.shape<1>
    %23 = hlfir.elemental %22 unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
    ^bb0(%arg0: index):
      %30 = hlfir.apply %20, %arg0 : (!hlfir.expr<?xi64>, index) -> i64
      %31 = fir.convert %30 : (i64) -> i32
      hlfir.yield_element %31 : i32
    }
    hlfir.assign %23 to %14#0 : !hlfir.expr<?xi32>, !fir.ref<!fir.array<3xi32>>
    hlfir.destroy %23 : !hlfir.expr<?xi32>
    hlfir.destroy %20 : !hlfir.expr<?xi64>
    %24 = mif.coshape coarray %11#0 : (!fir.ref<i32>) -> !fir.box<!fir.array<?xi64>>
    %25:2 = hlfir.declare %24 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.array<?xi64>>) -> (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>)
    %false_7 = arith.constant false
    %26 = hlfir.as_expr %25#0 move %false_7 : (!fir.box<!fir.array<?xi64>>, i1) -> !hlfir.expr<?xi64>
    %c0_8 = arith.constant 0 : index
    %27:3 = fir.box_dims %25#0, %c0_8 : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
    %28 = fir.shape %27#1 : (index) -> !fir.shape<1>
    %29 = hlfir.elemental %28 unordered : (!fir.shape<1>) -> !hlfir.expr<?xi64> {
    ^bb0(%arg0: index):
      %30 = hlfir.apply %26, %arg0 : (!hlfir.expr<?xi64>, index) -> i64
      hlfir.yield_element %30 : i64
    }
    hlfir.assign %29 to %17#0 : !hlfir.expr<?xi64>, !fir.ref<!fir.array<3xi64>>
    hlfir.destroy %29 : !hlfir.expr<?xi64>
    hlfir.destroy %26 : !hlfir.expr<?xi64>
    return
  }
}

// CHECK-LABEL: func.func @_QQmain
// CHECK: fir.call @_QMprifPprif_coshape
// CHECK: fir.call @_QMprifPprif_coshape
