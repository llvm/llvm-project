// RUN: fir-opt --fir-add-alias-tags --split-input-file %s | FileCheck %s

// This test demonstrates the need for fir.dummy_scope even
// when a function does not have dummy arguments.
//
// The original source is:
// module m
//   integer :: glob
// end module m
// subroutine test1
//   use m
//   call inner(glob)
//   glob = 2
// contains
//   subroutine inner(x)
//     integer :: x
//     integer :: y
//     y = 1
//     x = y
//   end subroutine inner
// end subroutine test1
//
// 'inner' function is manually inlined in FIR.
// When fir.dummy_scope is missing, TBAA tags for glob and x
// are placed into the same TBAA root. Since glob is a global
// and x is a dummy argument, TBAA ends up reporting no-alias
// for them, which is incorrect.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
func.func @_QPtest1() attributes {noinline} {
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %0 = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFtest1FinnerEy"}
  %1 = fir.address_of(@_QMmEglob) : !fir.ref<i32>
  %2 = fir.declare %1 {uniq_name = "_QMmEglob"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.dummy_scope : !fir.dscope
  %4 = fir.declare %2 dummy_scope %3 {uniq_name = "_QFtest1FinnerEx"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %5 = fir.declare %0 {uniq_name = "_QFtest1FinnerEy"} : (!fir.ref<i32>) -> !fir.ref<i32>
  fir.store %c1_i32 to %5 : !fir.ref<i32>
  %6 = fir.load %5 : !fir.ref<i32>
  fir.store %6 to %4 : !fir.ref<i32>
  fir.store %c2_i32 to %2 : !fir.ref<i32>
  return
}
}
// CHECK: #[[$ATTR_0:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest1">
// CHECK: #[[$ATTR_1:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[$ATTR_0]], 0>}>
// CHECK: #[[$ATTR_2:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[$ATTR_1]], 0>}>
// CHECK: #[[$TARGETDATA:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[$ATTR_2]], 0>}>
// CHECK: #[[$ATTR_3:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data", members = {<#[[$ATTR_2]], 0>}>
// CHECK: #[[$LOCAL_ATTR_0:.+]] = #llvm.tbaa_type_desc<id = "allocated data", members = {<#[[$TARGETDATA]], 0>}>
// CHECK: #[[$ATTR_5:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFtest1FinnerEx", members = {<#[[$ATTR_3]], 0>}>
// CHECK: #[[$ATTR_4:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[$TARGETDATA]], 0>}>
// CHECK: #[[$ATTR_7:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_5]], access_type = #[[$ATTR_5]], offset = 0>
// CHECK: #[[$LOCAL_ATTR_1:.+]] = #llvm.tbaa_type_desc<id = "allocated data/_QFtest1FinnerEy", members = {<#[[$LOCAL_ATTR_0]], 0>}>
// CHECK: #[[$ATTR_6:.+]] = #llvm.tbaa_type_desc<id = "global data/_QMmEglob", members = {<#[[$ATTR_4]], 0>}>
// CHECK: #[[$LOCAL_ATTR_2:.+]] = #llvm.tbaa_tag<base_type = #[[$LOCAL_ATTR_1]], access_type = #[[$LOCAL_ATTR_1]], offset = 0>
// CHECK: #[[$ATTR_8:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_6]], access_type = #[[$ATTR_6]], offset = 0>
// CHECK-LABEL:   func.func @_QPtest1() attributes {noinline} {
// CHECK:           %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFtest1FinnerEy"}
// CHECK:           %[[VAL_3:.*]] = fir.address_of(@_QMmEglob) : !fir.ref<i32>
// CHECK:           %[[VAL_4:.*]] = fir.declare %[[VAL_3]] {uniq_name = "_QMmEglob"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_5:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_6:.*]] = fir.declare %[[VAL_4]] dummy_scope %[[VAL_5]] {uniq_name = "_QFtest1FinnerEx"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[VAL_7:.*]] = fir.declare %[[VAL_2]] {uniq_name = "_QFtest1FinnerEy"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           fir.store %{{.*}} to %[[VAL_7]] {tbaa = [#[[$LOCAL_ATTR_2]]]} : !fir.ref<i32>
// CHECK:           %[[VAL_8:.*]] = fir.load %[[VAL_7]] {tbaa = [#[[$LOCAL_ATTR_2]]]}  : !fir.ref<i32>
// CHECK:           fir.store %[[VAL_8]] to %[[VAL_6]] {tbaa = [#[[$ATTR_7]]]} : !fir.ref<i32>
// CHECK:           fir.store %{{.*}} to %[[VAL_4]] {tbaa = [#[[$ATTR_8]]]} : !fir.ref<i32>

// -----

// This test has fir.dummy_scope in place, and TBAA is correct.
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
func.func @_QPtest2() attributes {noinline} {
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %0 = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFtest2FinnerEy"}
  %test_dummy_scope = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QMmEglob) : !fir.ref<i32>
  %2 = fir.declare %1 {uniq_name = "_QMmEglob"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.dummy_scope : !fir.dscope
  %4 = fir.declare %2 dummy_scope %3 {uniq_name = "_QFtest2FinnerEx"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %5 = fir.declare %0 {uniq_name = "_QFtest2FinnerEy"} : (!fir.ref<i32>) -> !fir.ref<i32>
  fir.store %c1_i32 to %5 : !fir.ref<i32>
  %6 = fir.load %5 : !fir.ref<i32>
  fir.store %6 to %4 : !fir.ref<i32>
  fir.store %c2_i32 to %2 : !fir.ref<i32>
  return
}
}
// CHECK: #[[$ATTR_0:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest2">
// CHECK: #[[$ATTR_1:.+]] = #llvm.tbaa_root<id = "Flang function root _QPtest2 - Scope 1">
// CHECK: #[[$ATTR_2:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[$ATTR_0]], 0>}>
// CHECK: #[[$ATTR_3:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[$ATTR_1]], 0>}>
// CHECK: #[[$ATTR_4:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[$ATTR_2]], 0>}>
// CHECK: #[[$ATTR_5:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[$ATTR_3]], 0>}>
// CHECK: #[[$TARGETDATA_0:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[$ATTR_4]], 0>}>
// CHECK: #[[$ATTR_6:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data", members = {<#[[$ATTR_5]], 0>}>
// CHECK: #[[$LOCAL_ATTR_0:.+]] = #llvm.tbaa_type_desc<id = "allocated data", members = {<#[[$TARGETDATA_0]], 0>}>
// CHECK: #[[$ATTR_8:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFtest2FinnerEx", members = {<#[[$ATTR_6]], 0>}>
// CHECK: #[[$ATTR_7:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[$TARGETDATA_0]], 0>}>
// CHECK: #[[$ATTR_10:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_8]], access_type = #[[$ATTR_8]], offset = 0>
// CHECK: #[[$LOCAL_ATTR_1:.+]] = #llvm.tbaa_type_desc<id = "allocated data/_QFtest2FinnerEy", members = {<#[[$LOCAL_ATTR_0]], 0>}>
// CHECK: #[[$ATTR_9:.+]] = #llvm.tbaa_type_desc<id = "global data/_QMmEglob", members = {<#[[$ATTR_7]], 0>}>
// CHECK: #[[$LOCAL_ATTR_2:.+]] = #llvm.tbaa_tag<base_type = #[[$LOCAL_ATTR_1]], access_type = #[[$LOCAL_ATTR_1]], offset = 0>
// CHECK: #[[$ATTR_11:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_9]], access_type = #[[$ATTR_9]], offset = 0>
// CHECK-LABEL:   func.func @_QPtest2() attributes {noinline} {
// CHECK:           %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "y", uniq_name = "_QFtest2FinnerEy"}
// CHECK:           %[[VAL_3:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_4:.*]] = fir.address_of(@_QMmEglob) : !fir.ref<i32>
// CHECK:           %[[VAL_5:.*]] = fir.declare %[[VAL_4]] {uniq_name = "_QMmEglob"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_7:.*]] = fir.declare %[[VAL_5]] dummy_scope %[[VAL_6]] {uniq_name = "_QFtest2FinnerEx"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
// CHECK:           %[[VAL_8:.*]] = fir.declare %[[VAL_2]] {uniq_name = "_QFtest2FinnerEy"} : (!fir.ref<i32>) -> !fir.ref<i32>
// CHECK:           fir.store %{{.*}} to %[[VAL_8]] {tbaa = [#[[$LOCAL_ATTR_2]]]}  : !fir.ref<i32>
// CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_8]] {tbaa = [#[[$LOCAL_ATTR_2]]]} : !fir.ref<i32>
// CHECK:           fir.store %[[VAL_9]] to %[[VAL_7]] {tbaa = [#[[$ATTR_10]]]} : !fir.ref<i32>
// CHECK:           fir.store %{{.*}} to %[[VAL_5]] {tbaa = [#[[$ATTR_11]]]} : !fir.ref<i32>
