// RUN: fir-opt --add-debug-info="emit-fake-use-for-debug-vars=true" %s | FileCheck %s --check-prefix=FAKE-USE
// RUN: fir-opt --add-debug-info="emit-fake-use-for-debug-vars=false" %s | FileCheck %s --check-prefix=NO-FAKE-USE

// Test that fir.fake_use ops are inserted for count and lower bound variables
// in each dimension of a multi-dimensional dynamic array.

// FAKE-USE-LABEL: func.func @test_2d_dynamic_array
// FAKE-USE: fircg.ext_declare %arg0(%[[COUNT1:.*]], %[[COUNT2:.*]]) origin %[[LB1:.*]], %[[LB2:.*]] dummy_scope
// FAKE-USE: fir.call @foo() : () -> ()
// FAKE-USE: fir.fake_use %arg0
// FAKE-USE: fir.fake_use %[[COUNT1]]
// FAKE-USE: fir.fake_use %[[COUNT2]]
// FAKE-USE: fir.fake_use %[[LB1]]
// FAKE-USE: fir.fake_use %[[LB2]]
// FAKE-USE: return

// NO-FAKE-USE-LABEL: func.func @test_2d_dynamic_array
// NO-FAKE-USE: fircg.ext_declare %arg0(%{{.*}}, %{{.*}}) origin %{{.*}}, %{{.*}} dummy_scope
// NO-FAKE-USE: fir.call @foo() : () -> ()
// NO-FAKE-USE-NOT: fir.fake_use
// NO-FAKE-USE: return

// FAKE-USE-LABEL: func.func @test_3d_dynamic_array
// FAKE-USE: fircg.ext_declare %arg0(%[[C1:.*]], %[[C2:.*]], %[[C3:.*]]) origin %[[L1:.*]], %[[L2:.*]], %[[L3:.*]] dummy_scope
// FAKE-USE: fir.call @foo() : () -> ()
// FAKE-USE: fir.fake_use %arg0
// FAKE-USE: fir.fake_use %[[C1]]
// FAKE-USE: fir.fake_use %[[C2]]
// FAKE-USE: fir.fake_use %[[C3]]
// FAKE-USE: fir.fake_use %[[L1]]
// FAKE-USE: fir.fake_use %[[L2]]
// FAKE-USE: fir.fake_use %[[L3]]
// FAKE-USE: return

// NO-FAKE-USE-LABEL: func.func @test_3d_dynamic_array
// NO-FAKE-USE: fircg.ext_declare %arg0(%{{.*}}, %{{.*}}, %{{.*}}) origin %{{.*}}, %{{.*}}, %{{.*}} dummy_scope
// NO-FAKE-USE: fir.call @foo() : () -> ()
// NO-FAKE-USE-NOT: fir.fake_use
// NO-FAKE-USE: return

// FAKE-USE-LABEL: func.func @test_2d_dynamic_array_multi_ret
// FAKE-USE: fircg.ext_declare %arg0(%[[MR_COUNT1:.*]], %[[MR_COUNT2:.*]]) origin %[[MR_LB1:.*]], %[[MR_LB2:.*]] dummy_scope
// FAKE-USE: cf.cond_br %{{.*}}, ^bb1, ^bb2
// FAKE-USE: ^bb1:
// FAKE-USE: fir.fake_use %arg0
// FAKE-USE: fir.fake_use %[[MR_COUNT1]]
// FAKE-USE: fir.fake_use %[[MR_COUNT2]]
// FAKE-USE: fir.fake_use %[[MR_LB1]]
// FAKE-USE: fir.fake_use %[[MR_LB2]]
// FAKE-USE: return
// FAKE-USE: ^bb2:
// FAKE-USE: fir.fake_use %arg0
// FAKE-USE: fir.fake_use %[[MR_COUNT1]]
// FAKE-USE: fir.fake_use %[[MR_COUNT2]]
// FAKE-USE: fir.fake_use %[[MR_LB1]]
// FAKE-USE: fir.fake_use %[[MR_LB2]]
// FAKE-USE: return

// NO-FAKE-USE-LABEL: func.func @test_2d_dynamic_array_multi_ret
// NO-FAKE-USE: fircg.ext_declare %arg0(%{{.*}}, %{{.*}}) origin %{{.*}}, %{{.*}} dummy_scope
// NO-FAKE-USE: cf.cond_br %{{.*}}, ^bb1, ^bb2
// NO-FAKE-USE: ^bb1:
// NO-FAKE-USE-NOT: fir.fake_use
// NO-FAKE-USE: return
// NO-FAKE-USE: ^bb2:
// NO-FAKE-USE-NOT: fir.fake_use
// NO-FAKE-USE: return

#loc1 = loc("debug-fake-use-multiple-dimensions.f90":1:1)
#loc3 = loc("debug-fake-use-multiple-dimensions.f90":3:14)
#loc4 = loc("debug-fake-use-multiple-dimensions.f90":4:14)
#loc5 = loc("debug-fake-use-multiple-dimensions.f90":5:1)
#loc6 = loc("debug-fake-use-multiple-dimensions.f90":6:1)
#loc = loc("debug-fake-use-multiple-dimensions.f90":0:0)

module attributes {dlti.dl_spec = #dlti.dl_spec<i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i32 = dense<32> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, i8 = dense<8> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, "dlti.stack_alignment" = 128 : i64, "dlti.mangling_mode" = "e", "dlti.endianness" = "little">, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", fir.target_cpu = "x86-64", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
  func.func private @foo()

  // 2D dynamically-sized array with a single return.
  func.func @test_2d_dynamic_array(%arg0: !fir.ref<!fir.array<?x?xi32>> {fir.bindc_name = "arr"} loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg1: !fir.ref<i32> {fir.bindc_name = "n1"} loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg2: !fir.ref<i32> {fir.bindc_name = "lb1"} loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg3: !fir.ref<i32> {fir.bindc_name = "n2"} loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg4: !fir.ref<i32> {fir.bindc_name = "lb2"} loc("debug-fake-use-multiple-dimensions.f90":1:1)) attributes {fir.internal_name = "_QPtest_2d_dynamic_array"} {
    %c0 = arith.constant 0 : index
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtest_2d_dynamic_arrayEn1"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc3)
    %2 = fircg.ext_declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFtest_2d_dynamic_arrayElb1"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc3)
    %3 = fircg.ext_declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFtest_2d_dynamic_arrayEn2"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc4)
    %4 = fircg.ext_declare %arg4 dummy_scope %0 arg 5 {uniq_name = "_QFtest_2d_dynamic_arrayElb2"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc4)
    %5 = fir.load %1 : !fir.ref<i32>
    %6 = fir.convert %5 : (i32) -> index
    %7 = arith.cmpi sgt, %6, %c0 : index
    %8 = arith.select %7, %6, %c0 : index
    %9 = fir.load %2 : !fir.ref<i32>
    %10 = fir.convert %9 : (i32) -> index
    %11 = fir.load %3 : !fir.ref<i32>
    %12 = fir.convert %11 : (i32) -> index
    %13 = arith.cmpi sgt, %12, %c0 : index
    %14 = arith.select %13, %12, %c0 : index
    %15 = fir.load %4 : !fir.ref<i32>
    %16 = fir.convert %15 : (i32) -> index
    %17 = fircg.ext_declare %arg0(%8, %14) origin %10, %16 dummy_scope %0 arg 1 {uniq_name = "_QFtest_2d_dynamic_arrayEarr"} : (!fir.ref<!fir.array<?x?xi32>>, index, index, index, index, !fir.dscope) -> !fir.ref<!fir.array<?x?xi32>> loc(#loc3)
    fir.call @foo() : () -> ()
    return loc(#loc5)
  } loc(#loc1)

  // 3D dynamically-sized array with a single return.
  func.func @test_3d_dynamic_array(%arg0: !fir.ref<!fir.array<?x?x?xi32>> {fir.bindc_name = "arr"} loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg1: !fir.ref<i32> {fir.bindc_name = "n1"} loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg2: !fir.ref<i32> {fir.bindc_name = "lb1"} loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg3: !fir.ref<i32> {fir.bindc_name = "n2"} loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg4: !fir.ref<i32> {fir.bindc_name = "lb2"} loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg5: !fir.ref<i32> {fir.bindc_name = "n3"} loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg6: !fir.ref<i32> {fir.bindc_name = "lb3"} loc("debug-fake-use-multiple-dimensions.f90":1:1)) attributes {fir.internal_name = "_QPtest_3d_dynamic_array"} {
    %c0 = arith.constant 0 : index
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtest_3d_dynamic_arrayEn1"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc3)
    %2 = fircg.ext_declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFtest_3d_dynamic_arrayElb1"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc3)
    %3 = fircg.ext_declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFtest_3d_dynamic_arrayEn2"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc4)
    %4 = fircg.ext_declare %arg4 dummy_scope %0 arg 5 {uniq_name = "_QFtest_3d_dynamic_arrayElb2"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc4)
    %5 = fircg.ext_declare %arg5 dummy_scope %0 arg 6 {uniq_name = "_QFtest_3d_dynamic_arrayEn3"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc3)
    %6 = fircg.ext_declare %arg6 dummy_scope %0 arg 7 {uniq_name = "_QFtest_3d_dynamic_arrayElb3"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc3)
    %7 = fir.load %1 : !fir.ref<i32>
    %8 = fir.convert %7 : (i32) -> index
    %9 = arith.cmpi sgt, %8, %c0 : index
    %10 = arith.select %9, %8, %c0 : index
    %11 = fir.load %2 : !fir.ref<i32>
    %12 = fir.convert %11 : (i32) -> index
    %13 = fir.load %3 : !fir.ref<i32>
    %14 = fir.convert %13 : (i32) -> index
    %15 = arith.cmpi sgt, %14, %c0 : index
    %16 = arith.select %15, %14, %c0 : index
    %17 = fir.load %4 : !fir.ref<i32>
    %18 = fir.convert %17 : (i32) -> index
    %19 = fir.load %5 : !fir.ref<i32>
    %20 = fir.convert %19 : (i32) -> index
    %21 = arith.cmpi sgt, %20, %c0 : index
    %22 = arith.select %21, %20, %c0 : index
    %23 = fir.load %6 : !fir.ref<i32>
    %24 = fir.convert %23 : (i32) -> index
    %25 = fircg.ext_declare %arg0(%10, %16, %22) origin %12, %18, %24 dummy_scope %0 arg 1 {uniq_name = "_QFtest_3d_dynamic_arrayEarr"} : (!fir.ref<!fir.array<?x?x?xi32>>, index, index, index, index, index, index, !fir.dscope) -> !fir.ref<!fir.array<?x?x?xi32>> loc(#loc3)
    fir.call @foo() : () -> ()
    return loc(#loc5)
  } loc(#loc1)

  // 2D dynamically-sized array with multiple returns.
  func.func @test_2d_dynamic_array_multi_ret(%arg0: !fir.ref<!fir.array<?x?xi32>> {fir.bindc_name = "arr"} loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg1: !fir.ref<i32> {fir.bindc_name = "n1"} loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg2: !fir.ref<i32> {fir.bindc_name = "lb1"} loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg3: !fir.ref<i32> {fir.bindc_name = "n2"} loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg4: !fir.ref<i32> {fir.bindc_name = "lb2"} loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg5: i1 loc("debug-fake-use-multiple-dimensions.f90":1:1)) attributes {fir.internal_name = "_QPtest_2d_dynamic_array_multi_ret"} {
    %c0 = arith.constant 0 : index
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtest_2d_dynamic_array_multi_retEn1"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc3)
    %2 = fircg.ext_declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFtest_2d_dynamic_array_multi_retElb1"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc3)
    %3 = fircg.ext_declare %arg3 dummy_scope %0 arg 4 {uniq_name = "_QFtest_2d_dynamic_array_multi_retEn2"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc4)
    %4 = fircg.ext_declare %arg4 dummy_scope %0 arg 5 {uniq_name = "_QFtest_2d_dynamic_array_multi_retElb2"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc4)
    %5 = fir.load %1 : !fir.ref<i32>
    %6 = fir.convert %5 : (i32) -> index
    %7 = arith.cmpi sgt, %6, %c0 : index
    %8 = arith.select %7, %6, %c0 : index
    %9 = fir.load %2 : !fir.ref<i32>
    %10 = fir.convert %9 : (i32) -> index
    %11 = fir.load %3 : !fir.ref<i32>
    %12 = fir.convert %11 : (i32) -> index
    %13 = arith.cmpi sgt, %12, %c0 : index
    %14 = arith.select %13, %12, %c0 : index
    %15 = fir.load %4 : !fir.ref<i32>
    %16 = fir.convert %15 : (i32) -> index
    %17 = fircg.ext_declare %arg0(%8, %14) origin %10, %16 dummy_scope %0 arg 1 {uniq_name = "_QFtest_2d_dynamic_array_multi_retEarr"} : (!fir.ref<!fir.array<?x?xi32>>, index, index, index, index, !fir.dscope) -> !fir.ref<!fir.array<?x?xi32>> loc(#loc3)
    fir.call @foo() : () -> ()
    cf.cond_br %arg5, ^bb1, ^bb2
  ^bb1:
    return loc(#loc5)
  ^bb2:
    return loc(#loc6)
  } loc(#loc1)
} loc(#loc)
