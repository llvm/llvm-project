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

module attributes {dlti.dl_spec = #dlti.dl_spec<i128 = dense<128> : vector<2xi64>, f80 = dense<128> : vector<2xi64>, i1 = dense<8> : vector<2xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i64 = dense<64> : vector<2xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i32 = dense<32> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, !llvm.ptr<270> = dense<32> : vector<4xi64>, i8 = dense<8> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, "dlti.stack_alignment" = 128 : i64, "dlti.mangling_mode" = "e", "dlti.endianness" = "little">, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", fir.target_cpu = "x86-64", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.ident = "flang", llvm.target_triple = "x86_64-unknown-linux-gnu"} {
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

  func.func @test_dynamic_array(%arg0: !fir.ref<!fir.array<?xi32>> {fir.bindc_name = "arr"} loc("debug-fake-use.f90":1:1), %arg1: !fir.ref<i32> {fir.bindc_name = "n"} loc("debug-fake-use.f90":1:1), %arg2: !fir.ref<i32> {fir.bindc_name = "lb"} loc("debug-fake-use.f90":1:1)) attributes {fir.internal_name = "_QPtest_dynamic_array"} {
    %c0 = arith.constant 0 : index
    %0 = fir.undefined !fir.dscope loc(#loc1)
    %1 = fircg.ext_declare %arg1 dummy_scope %0 arg 2 {uniq_name = "_QFtest_dynamic_arrayEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc3)
    %2 = fircg.ext_declare %arg2 dummy_scope %0 arg 3 {uniq_name = "_QFtest_dynamic_arrayElb"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32> loc(#loc4)
    %3 = fir.load %1 : !fir.ref<i32>
    %4 = fir.convert %3 : (i32) -> index
    %5 = arith.cmpi sgt, %4, %c0 : index
    %6 = arith.select %5, %4, %c0 : index
    %7 = fir.load %2 : !fir.ref<i32>
    %8 = fir.convert %7 : (i32) -> index
    %9 = fircg.ext_declare %arg0(%6) origin %8 dummy_scope %0 arg 1 {uniq_name = "_QFtest_dynamic_arrayEarr"} : (!fir.ref<!fir.array<?xi32>>, index, index, !fir.dscope) -> !fir.ref<!fir.array<?xi32>> loc(#loc3)
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
