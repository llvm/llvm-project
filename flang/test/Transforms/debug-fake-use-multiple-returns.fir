// RUN: fir-opt --add-debug-info="emit-fake-use-for-debug-vars=true" %s | FileCheck %s --check-prefix=FAKE-USE
// RUN: fir-opt --add-debug-info="emit-fake-use-for-debug-vars=false" %s | FileCheck %s --check-prefix=NO-FAKE-USE

// Test that fir.fake_use ops are inserted before every return in functions with
// multiple return locations.

// FAKE-USE-LABEL: func.func @test_dummy_arg_multi_ret
// FAKE-USE: %[[UNDEF:.*]] = fir.undefined !fir.dscope
// FAKE-USE: %[[DECL:.*]] = fircg.ext_declare %arg0 dummy_scope %[[UNDEF]] arg 1
// FAKE-USE: cf.cond_br %{{.*}}, ^bb1, ^bb2
// FAKE-USE: ^bb1:
// FAKE-USE: fir.fake_use %arg0
// FAKE-USE: return
// FAKE-USE: ^bb2:
// FAKE-USE: fir.fake_use %arg0
// FAKE-USE: return

// NO-FAKE-USE-LABEL: func.func @test_dummy_arg_multi_ret
// NO-FAKE-USE: %[[UNDEF:.*]] = fir.undefined !fir.dscope
// NO-FAKE-USE: %[[DECL:.*]] = fircg.ext_declare %arg0 dummy_scope %[[UNDEF]] arg 1
// NO-FAKE-USE: cf.cond_br %{{.*}}, ^bb1, ^bb2
// NO-FAKE-USE: ^bb1:
// NO-FAKE-USE-NOT: fir.fake_use
// NO-FAKE-USE: return
// NO-FAKE-USE: ^bb2:
// NO-FAKE-USE-NOT: fir.fake_use
// NO-FAKE-USE: return

// FAKE-USE-LABEL: func.func @test_dynamic_array_multi_ret
// FAKE-USE: fircg.ext_declare %arg0(%[[COUNT:.*]]) origin %[[LB:.*]] dummy_scope
// FAKE-USE: cf.cond_br %{{.*}}, ^bb1, ^bb2
// FAKE-USE: ^bb1:
// FAKE-USE: fir.fake_use %[[COUNT]]
// FAKE-USE: fir.fake_use %[[LB]]
// FAKE-USE: return
// FAKE-USE: ^bb2:
// FAKE-USE: fir.fake_use %[[COUNT]]
// FAKE-USE: fir.fake_use %[[LB]]
// FAKE-USE: return

// NO-FAKE-USE-LABEL: func.func @test_dynamic_array_multi_ret
// NO-FAKE-USE: fircg.ext_declare %arg0(%{{.*}}) origin %{{.*}} dummy_scope
// NO-FAKE-USE: cf.cond_br %{{.*}}, ^bb1, ^bb2
// NO-FAKE-USE: ^bb1:
// NO-FAKE-USE-NOT: fir.fake_use
// NO-FAKE-USE: return
// NO-FAKE-USE: ^bb2:
// NO-FAKE-USE-NOT: fir.fake_use
// NO-FAKE-USE: return

// FAKE-USE-LABEL: func.func @test_char_array_multi_ret
// FAKE-USE: fircg.ext_declare %arg0 typeparams %[[LEN:.*]] dummy_scope
// FAKE-USE: cf.cond_br %{{.*}}, ^bb1, ^bb2
// FAKE-USE: ^bb1:
// FAKE-USE: fir.fake_use %[[LEN]]
// FAKE-USE: return
// FAKE-USE: ^bb2:
// FAKE-USE: fir.fake_use %[[LEN]]
// FAKE-USE: return

// NO-FAKE-USE-LABEL: func.func @test_char_array_multi_ret
// NO-FAKE-USE: fircg.ext_declare %arg0 typeparams %{{.*}} dummy_scope
// NO-FAKE-USE: cf.cond_br %{{.*}}, ^bb1, ^bb2
// NO-FAKE-USE: ^bb1:
// NO-FAKE-USE-NOT: fir.fake_use
// NO-FAKE-USE: return
// NO-FAKE-USE: ^bb2:
// NO-FAKE-USE-NOT: fir.fake_use
// NO-FAKE-USE: return

#loc1 = loc("debug-fake-use-multiple-returns.f90":1:1)
#loc3 = loc("debug-fake-use-multiple-returns.f90":3:14)
#loc4 = loc("debug-fake-use-multiple-returns.f90":4:14)
#loc5 = loc("debug-fake-use-multiple-returns.f90":5:1)
#loc6 = loc("debug-fake-use-multiple-returns.f90":6:1)
#loc = loc("debug-fake-use-multiple-returns.f90":0:0)

module {
  func.func private @foo()

  // Function with a dummy argument and multiple returns.
  func.func @test_dummy_arg_multi_ret(%arg0: !fir.ref<i32> {fir.bindc_name = "expected"} loc("debug-fake-use-multiple-returns.f90":1:1), %arg1: i1 loc("debug-fake-use-multiple-returns.f90":1:1)) attributes {fir.internal_name = "_QPtest_dummy_arg_multi_ret"} {
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtest_dummy_arg_multi_retEexpected"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc3)
    fir.call @foo() : () -> ()
    cf.cond_br %arg1, ^bb1, ^bb2
  ^bb1:
    return loc(#loc5)
  ^bb2:
    return loc(#loc6)
  } loc(#loc1)

  // Function with a dynamically-sized array and multiple returns.
  func.func @test_dynamic_array_multi_ret(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "arr"} loc("debug-fake-use-multiple-returns.f90":1:1), %arg1: index loc("debug-fake-use-multiple-returns.f90":1:1), %arg2: index loc("debug-fake-use-multiple-returns.f90":1:1), %arg3: i1 loc("debug-fake-use-multiple-returns.f90":1:1)) attributes {fir.internal_name = "_QPtest_dynamic_array_multi_ret"} {
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg0(%arg1) origin %arg2 dummy_scope %0 arg 1 {uniq_name = "_QFtest_dynamic_array_multi_retEarr"} : (!fir.ref<!fir.array<?xi32>>, index, index, !fir.dscope) -> !fir.ref<!fir.array<?xi32>> loc(#loc3)
    fir.call @foo() : () -> ()
    cf.cond_br %arg3, ^bb1, ^bb2
  ^bb1:
    return loc(#loc5)
  ^bb2:
    return loc(#loc6)
  } loc(#loc1)

  // Function with a dynamically-sized character array and multiple returns.
  func.func @test_char_array_multi_ret(%arg0: !fir.ref<!fir.char<1,?>> {fir.bindc_name = "str"} loc("debug-fake-use-multiple-returns.f90":1:1), %arg1: index {fir.bindc_name = "n"} loc("debug-fake-use-multiple-returns.f90":1:1), %arg2: i1 loc("debug-fake-use-multiple-returns.f90":1:1)) attributes {fir.internal_name = "_QPtest_char_array_multi_ret"} {
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg0 typeparams %arg1 dummy_scope %0 arg 1 {uniq_name = "_QFtest_char_array_multi_retEstr"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> !fir.ref<!fir.char<1,?>> loc(#loc3)
    fir.call @foo() : () -> ()
    cf.cond_br %arg2, ^bb1, ^bb2
  ^bb1:
    return loc(#loc5)
  ^bb2:
    return loc(#loc6)
  } loc(#loc1)
} loc(#loc)
