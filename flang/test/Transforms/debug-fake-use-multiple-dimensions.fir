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

// FAKE-USE-LABEL: func.func @test_3d_mixed_array
// FAKE-USE: %[[CST_COUNT:.*]] = arith.constant 10 : index
// FAKE-USE: %[[CST_LB:.*]] = arith.constant 1 : index
// FAKE-USE: fircg.ext_declare %arg0(%[[CST_COUNT]], %[[DYN_COUNT1:.*]], %[[DYN_COUNT2:.*]]) origin %[[CST_LB]], %[[DYN_LB1:.*]], %[[DYN_LB2:.*]] dummy_scope
// FAKE-USE: fir.call @foo() : () -> ()
// FAKE-USE: fir.fake_use %arg0
// FAKE-USE: fir.fake_use %[[DYN_COUNT1]]
// FAKE-USE: fir.fake_use %[[DYN_COUNT2]]
// FAKE-USE: fir.fake_use %[[DYN_LB1]]
// FAKE-USE: fir.fake_use %[[DYN_LB2]]
// FAKE-USE-NOT: fir.fake_use
// FAKE-USE: return

// NO-FAKE-USE-LABEL: func.func @test_3d_mixed_array
// NO-FAKE-USE: fircg.ext_declare %arg0(%{{.*}}, %{{.*}}, %{{.*}}) origin %{{.*}}, %{{.*}}, %{{.*}} dummy_scope
// NO-FAKE-USE: fir.call @foo() : () -> ()
// NO-FAKE-USE-NOT: fir.fake_use
// NO-FAKE-USE: return

#loc1 = loc("debug-fake-use-multiple-dimensions.f90":1:1)
#loc3 = loc("debug-fake-use-multiple-dimensions.f90":3:14)
#loc4 = loc("debug-fake-use-multiple-dimensions.f90":4:14)
#loc5 = loc("debug-fake-use-multiple-dimensions.f90":5:1)
#loc6 = loc("debug-fake-use-multiple-dimensions.f90":6:1)
#loc = loc("debug-fake-use-multiple-dimensions.f90":0:0)

module {
  func.func private @foo()

  // 2D dynamically-sized array with a single return.
  func.func @test_2d_dynamic_array(%arg0: !fir.ref<!fir.array<?x?xi32>> {fir.bindc_name = "arr"} loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg1: index loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg2: index loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg3: index loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg4: index loc("debug-fake-use-multiple-dimensions.f90":1:1)) attributes {fir.internal_name = "_QPtest_2d_dynamic_array"} {
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg0(%arg1, %arg3) origin %arg2, %arg4 dummy_scope %0 arg 1 {uniq_name = "_QFtest_2d_dynamic_arrayEarr"} : (!fir.ref<!fir.array<?x?xi32>>, index, index, index, index, !fir.dscope) -> !fir.ref<!fir.array<?x?xi32>> loc(#loc3)
    fir.call @foo() : () -> ()
    return loc(#loc5)
  } loc(#loc1)

  // 3D dynamically-sized array with a single return.
  func.func @test_3d_dynamic_array(%arg0: !fir.ref<!fir.array<?x?x?xi32>> {fir.bindc_name = "arr"} loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg1: index loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg2: index loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg3: index loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg4: index loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg5: index loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg6: index loc("debug-fake-use-multiple-dimensions.f90":1:1)) attributes {fir.internal_name = "_QPtest_3d_dynamic_array"} {
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg0(%arg1, %arg3, %arg5) origin %arg2, %arg4, %arg6 dummy_scope %0 arg 1 {uniq_name = "_QFtest_3d_dynamic_arrayEarr"} : (!fir.ref<!fir.array<?x?x?xi32>>, index, index, index, index, index, index, !fir.dscope) -> !fir.ref<!fir.array<?x?x?xi32>> loc(#loc3)
    fir.call @foo() : () -> ()
    return loc(#loc5)
  } loc(#loc1)

  // 2D dynamically-sized array with multiple returns.
  func.func @test_2d_dynamic_array_multi_ret(%arg0: !fir.ref<!fir.array<?x?xi32>> {fir.bindc_name = "arr"} loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg1: index loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg2: index loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg3: index loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg4: index loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg5: i1 loc("debug-fake-use-multiple-dimensions.f90":1:1)) attributes {fir.internal_name = "_QPtest_2d_dynamic_array_multi_ret"} {
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg0(%arg1, %arg3) origin %arg2, %arg4 dummy_scope %0 arg 1 {uniq_name = "_QFtest_2d_dynamic_array_multi_retEarr"} : (!fir.ref<!fir.array<?x?xi32>>, index, index, index, index, !fir.dscope) -> !fir.ref<!fir.array<?x?xi32>> loc(#loc3)
    fir.call @foo() : () -> ()
    cf.cond_br %arg5, ^bb1, ^bb2
  ^bb1:
    return loc(#loc5)
  ^bb2:
    return loc(#loc6)
  } loc(#loc1)

  // 3D array with mixed constant and dynamic dimensions.
  func.func @test_3d_mixed_array(%arg0: !fir.ref<!fir.array<10x?x?xi32>> {fir.bindc_name = "arr"} loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg1: index loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg2: index loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg3: index loc("debug-fake-use-multiple-dimensions.f90":1:1), %arg4: index loc("debug-fake-use-multiple-dimensions.f90":1:1)) attributes {fir.internal_name = "_QPtest_3d_mixed_array"} {
    %c10 = arith.constant 10 : index
    %c1 = arith.constant 1 : index
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg0(%c10, %arg1, %arg3) origin %c1, %arg2, %arg4 dummy_scope %0 arg 1 {uniq_name = "_QFtest_3d_mixed_arrayEarr"} : (!fir.ref<!fir.array<10x?x?xi32>>, index, index, index, index, index, index, !fir.dscope) -> !fir.ref<!fir.array<10x?x?xi32>> loc(#loc3)
    fir.call @foo() : () -> ()
    return loc(#loc5)
  } loc(#loc1)
} loc(#loc)
