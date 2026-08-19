// Test that fir-add-alias-tags pass does not crash when it cannot find a name
// for a local target allocation. This can happen after compiling code with
// TARGET character results with `-mllvm -inline-all` where the allocation for
// the result is an unnamed temp that is given the target attribute inside the
// function being called.

// RUN: fir-opt --fir-add-alias-tags %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i8 = dense<[8, 32]> : vector<2xi64>, i16 = dense<[16, 32]> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 32, 64>, "dlti.stack_alignment" = 128 : i64, "dlti.function_pointer_alignment" = #dlti.function_pointer_alignment<32, function_dependent = true>>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"} {
  func.func @_QMmPfoo(%arg0: !fir.boxchar<1> {fir.bindc_name = "x"}, %arg1: !fir.boxchar<1> {fir.bindc_name = "c"}) {
    %c1 = arith.constant 1 : index
    %0 = fir.alloca !fir.char<1> {bindc_name = ".result"}
    %1 = fir.dummy_scope : !fir.dscope
    %9 = fir.dummy_scope : !fir.dscope
    %10 = fir.declare %0 typeparams %c1 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMmFbarEbar"} : (!fir.ref<!fir.char<1>>, index) -> !fir.ref<!fir.char<1>>
    %11 = fir.address_of(@_QQclX61) : !fir.ref<!fir.char<1>>
    %12 = fir.declare %11 typeparams %c1 {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQclX61"} : (!fir.ref<!fir.char<1>>, index) -> !fir.ref<!fir.char<1>>
    %13 = fir.load %12 : !fir.ref<!fir.char<1>>
    fir.store %13 to %10 : !fir.ref<!fir.char<1>>
    return
  }
  fir.global linkonce @_QQclX61 constant : !fir.char<1> {
    %0 = fir.string_lit "a"(1) : !fir.char<1>
    fir.has_value %0 : !fir.char<1>
  }
}


// CHECK: #[[TBAA_ROOT:.*]] = #llvm.tbaa_root<id = "Flang function root _QMmPfoo - Scope 1">
// CHECK: #[[TBAA_TYPE_DESC:.*]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[TBAA_ROOT]], 0>}>
// CHECK: #[[TBAA_TYPE_DESC2:.*]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[TBAA_TYPE_DESC]], 0>}>
// CHECK: #[[TBAA_TYPE_DESC4:.*]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[TBAA_TYPE_DESC2]], 0>}>
// CHECK: #[[TBAA_TYPE_DESC5:.*]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[TBAA_TYPE_DESC2]], 0>}>
// CHECK: #[[TBAA_TAG:.*]] = #llvm.tbaa_tag<base_type = #[[TBAA_TYPE_DESC5]], access_type = #[[TBAA_TYPE_DESC5]], offset = 0>

// CHECK-LABEL: func.func @_QMmPfoo
// CHECK: fir.store %{{.*}} to %{{.*}} {tbaa = [#[[TBAA_TAG]]]} : !fir.ref<!fir.char<1>>
