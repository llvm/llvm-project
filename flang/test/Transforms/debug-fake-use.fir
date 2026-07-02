// RUN: fir-opt --add-debug-info="emit-fake-use-for-debug-vars=true" %s | FileCheck %s --check-prefix=FAKE-USE
// RUN: fir-opt --add-debug-info="emit-fake-use-for-debug-vars=false" %s | FileCheck %s --check-prefix=NO-FAKE-USE

// FAKE-USE-LABEL: func.func @test_
// FAKE-USE: %[[UNDEF:.*]] = fir.undefined !fir.dscope
// FAKE-USE: %[[DECL:.*]] = fircg.ext_declare %arg0 dummy_scope %[[UNDEF]] arg 1
// FAKE-USE: fir.call @foo() : () -> ()
// FAKE-USE: fir.fake_use %arg0
// FAKE-USE: return

// NO-FAKE-USE-LABEL: func.func @test_
// NO-FAKE-USE: %[[UNDEF:.*]] = fir.undefined !fir.dscope
// NO-FAKE-USE: %[[DECL:.*]] = fircg.ext_declare %arg0 dummy_scope %[[UNDEF]] arg 1
// NO-FAKE-USE: fir.call @foo() : () -> ()
// NO-FAKE-USE-NOT: fir.fake_use
// NO-FAKE-USE: return

// FAKE-USE-LABEL: func.func @test_local
// FAKE-USE: %[[UNDEF:.*]] = fir.undefined !fir.dscope
// FAKE-USE: %[[ALLOCA:.*]] = fir.alloca i32
// FAKE-USE: %[[DECL_LOCAL:.*]] = fircg.ext_declare %[[ALLOCA]] dummy_scope %[[UNDEF]] {uniq_name = "_QFtestElocal"}
// FAKE-USE: fir.call @foo() : () -> ()
// FAKE-USE-NOT: fir.fake_use
// FAKE-USE: return

// NO-FAKE-USE-LABEL: func.func @test_local
// NO-FAKE-USE: %[[UNDEF:.*]] = fir.undefined !fir.dscope
// NO-FAKE-USE: %[[ALLOCA:.*]] = fir.alloca i32
// NO-FAKE-USE: %[[DECL_LOCAL:.*]] = fircg.ext_declare %[[ALLOCA]] dummy_scope %[[UNDEF]] {uniq_name = "_QFtestElocal"}
// NO-FAKE-USE: fir.call @foo() : () -> ()
// NO-FAKE-USE-NOT: fir.fake_use
// NO-FAKE-USE: return

// FAKE-USE-LABEL: func.func @test_dynamic_array
// FAKE-USE: fircg.ext_declare %arg0(%[[COUNT:.*]]) origin %[[LB:.*]] dummy_scope
// FAKE-USE: fir.call @foo() : () -> ()
// FAKE-USE: fir.fake_use %[[COUNT]]
// FAKE-USE: fir.fake_use %[[LB]]
// FAKE-USE: return

// NO-FAKE-USE-LABEL: func.func @test_dynamic_array
// NO-FAKE-USE: fircg.ext_declare %arg0(%{{.*}}) origin %{{.*}} dummy_scope
// NO-FAKE-USE: fir.call @foo() : () -> ()
// NO-FAKE-USE-NOT: fir.fake_use
// NO-FAKE-USE: return

// FAKE-USE-LABEL: func.func @test_char_array
// FAKE-USE: fircg.ext_declare %arg0 typeparams %[[LEN:.*]] dummy_scope
// FAKE-USE: fir.call @foo() : () -> ()
// FAKE-USE: fir.fake_use %[[LEN]]
// FAKE-USE: return

// NO-FAKE-USE-LABEL: func.func @test_char_array
// NO-FAKE-USE: fircg.ext_declare %arg0 typeparams %{{.*}} dummy_scope
// NO-FAKE-USE: fir.call @foo() : () -> ()
// NO-FAKE-USE-NOT: fir.fake_use
// NO-FAKE-USE: return

#loc1 = loc("debug-fake-use.f90":1:1)
#loc3 = loc("debug-fake-use.f90":3:14)
#loc4 = loc("debug-fake-use.f90":4:14)
#loc5 = loc("debug-fake-use.f90":5:1)
#loc = loc("debug-fake-use.f90":0:0)

module {
  func.func private @foo()
  func.func @test_(%arg0: !fir.ref<i32> {fir.bindc_name = "expected"} loc("debug-fake-use.f90":1:1)) attributes {fir.internal_name = "_QPtest"} {
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg0 dummy_scope %0 arg 1 {uniq_name = "_QFtestEexpected"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc3)
    fir.call @foo() : () -> ()
    return loc(#loc5)
  } loc(#loc1)

  func.func @test_local() attributes {fir.internal_name = "_QPtest_local"} {
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fir.alloca i32 loc(#loc4)
    %2 = fircg.ext_declare %1 dummy_scope %0 {uniq_name = "_QFtestElocal"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc4)
    fir.call @foo() : () -> ()
    return loc(#loc5)
  } loc(#loc1)

  func.func @test_dynamic_array(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "arr"} loc("debug-fake-use.f90":1:1), %arg1: index loc("debug-fake-use.f90":1:1), %arg2: index loc("debug-fake-use.f90":1:1)) attributes {fir.internal_name = "_QPtest_dynamic_array"} {
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg0(%arg1) origin %arg2 dummy_scope %0 arg 1 {uniq_name = "_QFtest_dynamic_arrayEarr"} : (!fir.ref<!fir.array<?xi32>>, index, index, !fir.dscope) -> !fir.ref<!fir.array<?xi32>> loc(#loc3)
    fir.call @foo() : () -> ()
    return loc(#loc5)
  } loc(#loc1)

  func.func @test_char_array(%arg0: !fir.ref<!fir.char<1,?>> {fir.bindc_name = "str"} loc("debug-fake-use.f90":1:1), %arg1: index {fir.bindc_name = "n"} loc("debug-fake-use.f90":1:1)) attributes {fir.internal_name = "_QPtest_char_array"} {
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg0 typeparams %arg1 dummy_scope %0 arg 1 {uniq_name = "_QFtest_char_arrayEstr"} : (!fir.ref<!fir.char<1,?>>, index, !fir.dscope) -> !fir.ref<!fir.char<1,?>> loc(#loc3)
    fir.call @foo() : () -> ()
    return loc(#loc5)
  } loc(#loc1)
} loc(#loc)
