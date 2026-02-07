// RUN: fir-opt -unsafe-cray-pointers --fir-add-alias-tags %s | FileCheck %s

// Fortran source:
// subroutine test()
//   real :: a, b
//   pointer(p, a)
//   p = loc(b)
//   b = 2
// end subroutine

// CHECK:      #[[TBAA_ROOT:.*]] = #llvm.tbaa_root<id = "Flang function root _QPtest">
// CHECK-NEXT: #[[ANY_ACCESS:.*]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[TBAA_ROOT]], 0>}>
// CHECK-NEXT: #[[ANY_DATA:.*]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[ANY_ACCESS]], 0>}>
// CHECK-NEXT: #[[ANY_DATA_TAG:.*]] = #llvm.tbaa_tag<base_type = #[[ANY_DATA]], access_type = #[[ANY_DATA]], offset = 0>
// CHECK-NEXT: #[[ALLOCATED_DATA:.*]] = #llvm.tbaa_type_desc<id = "allocated data", members = {<#[[ANY_DATA]], 0>}>
// CHECK-NEXT: #[[B:.*]] = #llvm.tbaa_type_desc<id = "allocated data/_QFtestEb", members = {<#[[ALLOCATED_DATA]], 0>}>
// CHECK-NEXT: #[[B_TAG:.*]] = #llvm.tbaa_tag<base_type = #[[B]], access_type = #[[B]], offset = 0>

module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr<270> = dense<32> : vector<4xi64>, !llvm.ptr<271> = dense<32> : vector<4xi64>, !llvm.ptr<272> = dense<64> : vector<4xi64>, i8 = dense<[8, 32]> : vector<2xi64>, i16 = dense<[16, 32]> : vector<2xi64>, i64 = dense<64> : vector<2xi64>, i128 = dense<128> : vector<2xi64>, !llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little", "dlti.mangling_mode" = "e", "dlti.legal_int_widths" = array<i32: 32, 64>, "dlti.stack_alignment" = 128 : i64, "dlti.function_pointer_alignment" = #dlti.function_pointer_alignment<32, function_dependent = true>>, fir.defaultkind = "a1c4d8i4l4r4", fir.kindmap = "", llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"} {
// CHECK-LABEL: func.func @_QPtest()
  func.func @_QPtest() {
    %cst = arith.constant 2.000000e+00 : f32
    %0 = fir.alloca !fir.box<!fir.ptr<f32>>
    %1 = fir.dummy_scope : !fir.dscope
    %2 = fir.alloca i64 {bindc_name = "p", uniq_name = "_QFtestEp"}
    %3 = fir.declare %2 {fortran_attrs = #fir.var_attrs<cray_pointer>, uniq_name = "_QFtestEp"} : (!fir.ref<i64>) -> !fir.ref<i64>
    %4 = fir.alloca f32 {bindc_name = "b", uniq_name = "_QFtestEb"}
    %5 = fir.declare %4 {uniq_name = "_QFtestEb"} : (!fir.ref<f32>) -> !fir.ref<f32>
    %6 = fir.declare %0 {fortran_attrs = #fir.var_attrs<pointer, cray_pointee>, uniq_name = "_QFtestEa"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
    %7 = fir.zero_bits !fir.ptr<f32>
    %8 = fir.embox %7 : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
    fir.store %8 to %6 : !fir.ref<!fir.box<!fir.ptr<f32>>>
// Descriptor tagged in codegen
// CHECK: fir.store %{{.*}} to %{{.*}} : !fir.ref<!fir.box<!fir.ptr<f32>>
    %9 = fir.convert %5 : (!fir.ref<f32>) -> i64
    fir.store %9 to %3 : !fir.ref<i64>
// CHECK: fir.store {{.*}} to {{.*}} {tbaa = [#[[ANY_DATA_TAG]]]} : !fir.ref<i64>
    fir.store %cst to %5 : !fir.ref<f32>
// CHECK: fir.store {{.*}} to {{.*}} {tbaa = [#[[B_TAG]]]} : !fir.ref<f32>
    return
  }
}

