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
    %11:2 = hlfir.declare %3 {uniq_name = "_QFEa"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %c3 = arith.constant 3 : index
    %12 = fir.alloca !fir.array<3xi32> {bindc_name = "res", uniq_name = "_QFEres"}
    %13 = fir.shape %c3 : (index) -> !fir.shape<1>
    %14:2 = hlfir.declare %12(%13) {uniq_name = "_QFEres"} : (!fir.ref<!fir.array<3xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi32>>, !fir.ref<!fir.array<3xi32>>)
    %c3_5 = arith.constant 3 : index
    %15 = fir.alloca !fir.array<3xi64> {bindc_name = "res2", uniq_name = "_QFEres2"}
    %16 = fir.shape %c3_5 : (index) -> !fir.shape<1>
    %17:2 = hlfir.declare %15(%16) {uniq_name = "_QFEres2"} : (!fir.ref<!fir.array<3xi64>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xi64>>, !fir.ref<!fir.array<3xi64>>)
    %18 = fir.embox %11#0 : (!fir.ref<i32>) -> !fir.box<i32, corank:3>
    %19 = mif.coshape coarray %18 : (!fir.box<i32, corank:3>) -> !fir.box<!fir.array<?xi64>>
    %20:2 = hlfir.declare %19 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.array<?xi64>>) -> (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>)
    %false = arith.constant false
    %21 = hlfir.as_expr %20#0 move %false : (!fir.box<!fir.array<?xi64>>, i1) -> !hlfir.expr<?xi64>
    %c0_6 = arith.constant 0 : index
    %22:3 = fir.box_dims %20#0, %c0_6 : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
    %23 = fir.shape %22#1 : (index) -> !fir.shape<1>
    %24 = hlfir.elemental %23 unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
    ^bb0(%arg0: index):
      %32 = hlfir.apply %21, %arg0 : (!hlfir.expr<?xi64>, index) -> i64
      %33 = fir.convert %32 : (i64) -> i32
      hlfir.yield_element %33 : i32
    }
    hlfir.assign %24 to %14#0 : !hlfir.expr<?xi32>, !fir.ref<!fir.array<3xi32>>
    hlfir.destroy %24 : !hlfir.expr<?xi32>
    hlfir.destroy %21 : !hlfir.expr<?xi64>
    %25 = fir.embox %11#0 : (!fir.ref<i32>) -> !fir.box<i32, corank:3>
    %26 = mif.coshape coarray %25 : (!fir.box<i32, corank:3>) -> !fir.box<!fir.array<?xi64>>
    %27:2 = hlfir.declare %26 {uniq_name = ".tmp.intrinsic_result"} : (!fir.box<!fir.array<?xi64>>) -> (!fir.box<!fir.array<?xi64>>, !fir.box<!fir.array<?xi64>>)
    %false_7 = arith.constant false
    %28 = hlfir.as_expr %27#0 move %false_7 : (!fir.box<!fir.array<?xi64>>, i1) -> !hlfir.expr<?xi64>
    %c0_8 = arith.constant 0 : index
    %29:3 = fir.box_dims %27#0, %c0_8 : (!fir.box<!fir.array<?xi64>>, index) -> (index, index, index)
    %30 = fir.shape %29#1 : (index) -> !fir.shape<1>
    %31 = hlfir.elemental %30 unordered : (!fir.shape<1>) -> !hlfir.expr<?xi64> {
    ^bb0(%arg0: index):
      %32 = hlfir.apply %28, %arg0 : (!hlfir.expr<?xi64>, index) -> i64
      hlfir.yield_element %32 : i64
    }
    hlfir.assign %31 to %17#0 : !hlfir.expr<?xi64>, !fir.ref<!fir.array<3xi64>>
    hlfir.destroy %31 : !hlfir.expr<?xi64>
    hlfir.destroy %28 : !hlfir.expr<?xi64>
    return
  }

  // Test that MIFCoshapeOpConversion emits an element-wise i64→i32 conversion
  // loop when the declared element type differs from the i64 written by
  // prif_coshape. Models `integer :: c[5,*]` (corank:2, default kind = i32).
  func.func @test_coshape_i32() {
    %c1_i64 = arith.constant 1 : i64
    %c5_i64 = arith.constant 5 : i64
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %lc = fir.alloca !fir.array<2xi64>
    %uc = fir.alloca !fir.array<1xi64>
    %lc0 = fir.coordinate_of %lc, %c0 : (!fir.ref<!fir.array<2xi64>>, index) -> !fir.ref<i64>
    fir.store %c1_i64 to %lc0 : !fir.ref<i64>
    %lc1 = fir.coordinate_of %lc, %c1 : (!fir.ref<!fir.array<2xi64>>, index) -> !fir.ref<i64>
    fir.store %c1_i64 to %lc1 : !fir.ref<i64>
    %uc0 = fir.coordinate_of %uc, %c0 : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
    fir.store %c5_i64 to %uc0 : !fir.ref<i64>
    %lb = fir.embox %lc : (!fir.ref<!fir.array<2xi64>>) -> !fir.box<!fir.array<2xi64>>
    %ub = fir.embox %uc : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
    %c_ref = fir.address_of(@_QFEc) : !fir.ref<i32>
    mif.alloc_coarray %c_ref lcobounds %lb ucobounds %ub {uniq_name = "_QFEc"} : (!fir.ref<i32>, !fir.box<!fir.array<2xi64>>, !fir.box<!fir.array<1xi64>>) -> ()
    %c_decl:2 = hlfir.declare %c_ref {uniq_name = "_QFEc"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %cobox = fir.embox %c_decl#0 : (!fir.ref<i32>) -> !fir.box<i32, corank:2>
    %res = mif.coshape coarray %cobox : (!fir.box<i32, corank:2>) -> !fir.box<!fir.array<?xi32>>
    return
  }
}

// CHECK-LABEL: func.func @_QQmain
// CHECK: fir.call @_QMprifPprif_coshape
// CHECK: fir.call @_QMprifPprif_coshape

// CHECK-LABEL: func.func @test_coshape_i32
// i32 and i64 scratch buffers are hoisted to the function entry block.
// CHECK-DAG:     %[[I32BUF:.*]] = fir.alloca !fir.array<2xi32>
// CHECK-DAG:     %[[I64BUF:.*]] = fir.alloca !fir.array<2xi64>
// prif_coshape writes i64 values into the i64 buffer.
// CHECK:         fir.call @_QMprifPprif_coshape
// After the call, `result` is converted to a dynamic-extent i64 box and
// then its base pointer is extracted for the per-element copy loop.
// CHECK:         %[[I64BOX:.*]] = fir.convert {{.*}} : (!fir.box<!fir.array<2xi64>>) -> !fir.box<!fir.array<?xi64>>
// CHECK:         %[[I64BUFREF:.*]] = fir.box_addr %[[I64BOX]] : (!fir.box<!fir.array<?xi64>>) -> !fir.ref<!fir.array<?xi64>>
// Loop iteration 0: load i64, truncate to i32, store into i32 buffer.
// CHECK:         %[[C0:.*]] = arith.constant 0 : index
// CHECK:         %[[SRC0:.*]] = fir.coordinate_of %[[I64BUFREF]], %[[C0]] : (!fir.ref<!fir.array<?xi64>>, index) -> !fir.ref<i64>
// CHECK:         %[[VAL0:.*]] = fir.load %[[SRC0]] : !fir.ref<i64>
// CHECK:         %[[CONV0:.*]] = fir.convert %[[VAL0]] : (i64) -> i32
// CHECK:         %[[DST0:.*]] = fir.coordinate_of %[[I32BUF]], %[[C0]] : (!fir.ref<!fir.array<2xi32>>, index) -> !fir.ref<i32>
// CHECK:         fir.store %[[CONV0]] to %[[DST0]] : !fir.ref<i32>
// Loop iteration 1: same pattern for the second codimension.
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         %[[SRC1:.*]] = fir.coordinate_of %[[I64BUFREF]], %[[C1]] : (!fir.ref<!fir.array<?xi64>>, index) -> !fir.ref<i64>
// CHECK:         %[[VAL1:.*]] = fir.load %[[SRC1]] : !fir.ref<i64>
// CHECK:         %[[CONV1:.*]] = fir.convert %[[VAL1]] : (i64) -> i32
// CHECK:         %[[DST1:.*]] = fir.coordinate_of %[[I32BUF]], %[[C1]] : (!fir.ref<!fir.array<2xi32>>, index) -> !fir.ref<i32>
// CHECK:         fir.store %[[CONV1]] to %[[DST1]] : !fir.ref<i32>
// The i32 buffer is boxed and widened to a dynamic-extent result.
// CHECK:         %[[BOX_I32:.*]] = fir.embox %[[I32BUF]] : (!fir.ref<!fir.array<2xi32>>) -> !fir.box<!fir.array<2xi32>>
// CHECK:         fir.convert %[[BOX_I32]] : (!fir.box<!fir.array<2xi32>>) -> !fir.box<!fir.array<?xi32>>
