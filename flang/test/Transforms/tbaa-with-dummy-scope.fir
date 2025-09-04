// RUN: fir-opt --fir-add-alias-tags --split-input-file %s | FileCheck %s

// subroutine test(x, y)
//   real, target :: x, y
//   x = y
//   call inner(x, y) ! the inlined load/store go to Scope 1
//   call inner(x, y) ! the inlined load/store go to Scope 2
// contains
//   subroutine inner(x, y)
//     real :: x, y
//     x = y
//   end subroutine inner
// end subroutine test

// CHECK: #[[TEST1ROOT:.+]] = #llvm.tbaa_root<id = "Flang function root test1">
// CHECK: #[[$ATTR_0:.+]] = #llvm.tbaa_root<id = "Flang function root test1 - Scope 1">
// CHECK: #[[$ATTR_1:.+]] = #llvm.tbaa_root<id = "Flang function root test1 - Scope 2">
// CHECK: #[[TEST1ANYACCESS:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[TEST1ROOT]], 0>}>
// CHECK: #[[$ATTR_2:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[$ATTR_0]], 0>}>
// CHECK: #[[$ATTR_3:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[$ATTR_1]], 0>}>
// CHECK: #[[TEST1ANYDATA:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[TEST1ANYACCESS]], 0>}>
// CHECK: #[[$ATTR_4:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[$ATTR_2]], 0>}>
// CHECK: #[[$ATTR_5:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[$ATTR_3]], 0>}>
// CHECK: #[[TARGETDATA:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[TEST1ANYDATA]], 0>}>
// CHECK: #[[$ATTR_6:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data", members = {<#[[$ATTR_4]], 0>}>
// CHECK: #[[$ATTR_7:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data", members = {<#[[$ATTR_5]], 0>}>
// CHECK: #[[TARGETTAG:.+]] = #llvm.tbaa_tag<base_type = #[[TARGETDATA]], access_type = #[[TARGETDATA]], offset = 0>
// CHECK: #[[$ATTR_8:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFtestFinnerEy", members = {<#[[$ATTR_6]], 0>}>
// CHECK: #[[$ATTR_9:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFtestFinnerEx", members = {<#[[$ATTR_6]], 0>}>
// CHECK: #[[$ATTR_10:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFtestFinnerEy", members = {<#[[$ATTR_7]], 0>}>
// CHECK: #[[$ATTR_11:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFtestFinnerEx", members = {<#[[$ATTR_7]], 0>}>
// CHECK: #[[$ATTR_12:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_8]], access_type = #[[$ATTR_8]], offset = 0>
// CHECK: #[[$ATTR_13:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_9]], access_type = #[[$ATTR_9]], offset = 0>
// CHECK: #[[$ATTR_14:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_10]], access_type = #[[$ATTR_10]], offset = 0>
// CHECK: #[[$ATTR_15:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_11]], access_type = #[[$ATTR_11]], offset = 0>
// CHECK:   func.func @test1(
// CHECK:           %[[VAL_5:.*]] = fir.load %{{.*}} {tbaa = [#[[TARGETTAG]]]} : !fir.ref<f32>
// CHECK:           fir.store %{{.*}} {tbaa = [#[[TARGETTAG]]]} : !fir.ref<f32>
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_9:.*]] = fir.load %{{.*}} {tbaa = [#[[$ATTR_12]]]} : !fir.ref<f32>
// CHECK:           fir.store %{{.*}} {tbaa = [#[[$ATTR_13]]]} : !fir.ref<f32>
// CHECK:           %[[VAL_10:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_13:.*]] = fir.load %{{.*}} {tbaa = [#[[$ATTR_14]]]} : !fir.ref<f32>
// CHECK:           fir.store %{{.*}} {tbaa = [#[[$ATTR_15]]]} : !fir.ref<f32>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
func.func @test1(%arg0: !fir.ref<f32> {fir.bindc_name = "x", fir.target}, %arg1: !fir.ref<f32> {fir.bindc_name = "y", fir.target}) {
  %scope_out = fir.dummy_scope : !fir.dscope
  %0 = fir.declare %arg0 dummy_scope %scope_out {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEx"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %1 = fir.declare %arg1 dummy_scope %scope_out {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEy"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %2 = fir.load %1 : !fir.ref<f32>
  fir.store %2 to %0 : !fir.ref<f32>
  %scope_in1 = fir.dummy_scope : !fir.dscope
  %3 = fir.declare %0 dummy_scope %scope_in1 {uniq_name = "_QFtestFinnerEx"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %4 = fir.declare %1 dummy_scope %scope_in1 {uniq_name = "_QFtestFinnerEy"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %5 = fir.load %4 : !fir.ref<f32>
  fir.store %5 to %3 : !fir.ref<f32>
  %scope_in2 = fir.dummy_scope : !fir.dscope
  %6 = fir.declare %0 dummy_scope %scope_in2 {uniq_name = "_QFtestFinnerEx"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %7 = fir.declare %1 dummy_scope %scope_in2 {uniq_name = "_QFtestFinnerEy"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %8 = fir.load %7 : !fir.ref<f32>
  fir.store %8 to %6 : !fir.ref<f32>
  return
}
}

// -----

// module test
//   real :: x, y
// contains
// subroutine caller(z)
//   real :: z
//   x = y ! the load/store go to the root scope
//   call callee(z)
// end subroutine caller
// subroutine callee(z)
//   x = y ! the load/store go to Scope 1
//   y = z ! the load/store go to Scope 1
// end subroutine callee
// end module test

// CHECK: #[[$ATTR_32:.+]] = #llvm.tbaa_root<id = "Flang function root _QMtestPcaller">
// CHECK: #[[$ATTR_33:.+]] = #llvm.tbaa_root<id = "Flang function root _QMtestPcaller - Scope 1">
// CHECK: #[[$ATTR_34:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[$ATTR_32]], 0>}>
// CHECK: #[[$ATTR_35:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[$ATTR_33]], 0>}>
// CHECK: #[[$ATTR_36:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[$ATTR_34]], 0>}>
// CHECK: #[[$ATTR_37:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[$ATTR_35]], 0>}>
// CHECK: #[[CALLERTARGETDATA:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[$ATTR_36]], 0>}>
// CHECK: #[[CALLEETARGETDATA:.+]] = #llvm.tbaa_type_desc<id = "target data", members = {<#[[$ATTR_37]], 0>}>
// CHECK: #[[$ATTR_40:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data", members = {<#[[$ATTR_37]], 0>}>
// CHECK: #[[$ATTR_38:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[CALLERTARGETDATA]], 0>}>
// CHECK: #[[$ATTR_39:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#[[CALLEETARGETDATA]], 0>}>
// CHECK: #[[$ATTR_45:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QMtestFcalleeEz", members = {<#[[$ATTR_40]], 0>}>
// CHECK: #[[$ATTR_50:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_45]], access_type = #[[$ATTR_45]], offset = 0>
// CHECK: #[[$ATTR_41:.+]] = #llvm.tbaa_type_desc<id = "global data/_QMtestEy", members = {<#[[$ATTR_38]], 0>}>
// CHECK: #[[$ATTR_42:.+]] = #llvm.tbaa_type_desc<id = "global data/_QMtestEx", members = {<#[[$ATTR_38]], 0>}>
// CHECK: #[[$ATTR_43:.+]] = #llvm.tbaa_type_desc<id = "global data/_QMtestEy", members = {<#[[$ATTR_39]], 0>}>
// CHECK: #[[$ATTR_44:.+]] = #llvm.tbaa_type_desc<id = "global data/_QMtestEx", members = {<#[[$ATTR_39]], 0>}>
// CHECK: #[[$ATTR_46:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_41]], access_type = #[[$ATTR_41]], offset = 0>
// CHECK: #[[$ATTR_47:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_42]], access_type = #[[$ATTR_42]], offset = 0>
// CHECK: #[[$ATTR_48:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_43]], access_type = #[[$ATTR_43]], offset = 0>
// CHECK: #[[$ATTR_49:.+]] = #llvm.tbaa_tag<base_type = #[[$ATTR_44]], access_type = #[[$ATTR_44]], offset = 0>
// CHECK:   func.func @_QMtestPcaller(
// CHECK-SAME:                              %[[VAL_0:.*]]: !fir.ref<f32> {fir.bindc_name = "z"}) {
// CHECK:           %[[VAL_1:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_2:.*]] = fir.address_of(@_QMtestEx) : !fir.ref<f32>
// CHECK:           %[[VAL_3:.*]] = fir.declare %[[VAL_2]] {uniq_name = "_QMtestEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
// CHECK:           %[[VAL_4:.*]] = fir.address_of(@_QMtestEy) : !fir.ref<f32>
// CHECK:           %[[VAL_5:.*]] = fir.declare %[[VAL_4]] {uniq_name = "_QMtestEy"} : (!fir.ref<f32>) -> !fir.ref<f32>
// CHECK:           %[[VAL_6:.*]] = fir.declare %[[VAL_0]] dummy_scope %[[VAL_1]] {uniq_name = "_QMtestFcallerEz"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
// CHECK:           %[[VAL_7:.*]] = fir.load %[[VAL_5]] {tbaa = [#[[$ATTR_46]]]} : !fir.ref<f32>
// CHECK:           fir.store %[[VAL_7]] to %[[VAL_3]] {tbaa = [#[[$ATTR_47]]]} : !fir.ref<f32>
// CHECK:           %[[VAL_8:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_9:.*]] = fir.address_of(@_QMtestEx) : !fir.ref<f32>
// CHECK:           %[[VAL_10:.*]] = fir.declare %[[VAL_9]] {uniq_name = "_QMtestEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
// CHECK:           %[[VAL_11:.*]] = fir.address_of(@_QMtestEy) : !fir.ref<f32>
// CHECK:           %[[VAL_12:.*]] = fir.declare %[[VAL_11]] {uniq_name = "_QMtestEy"} : (!fir.ref<f32>) -> !fir.ref<f32>
// CHECK:           %[[VAL_13:.*]] = fir.declare %[[VAL_6]] dummy_scope %[[VAL_8]] {uniq_name = "_QMtestFcalleeEz"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
// CHECK:           %[[VAL_14:.*]] = fir.load %[[VAL_12]] {tbaa = [#[[$ATTR_48]]]} : !fir.ref<f32>
// CHECK:           fir.store %[[VAL_14]] to %[[VAL_10]] {tbaa = [#[[$ATTR_49]]]} : !fir.ref<f32>
// CHECK:           %[[VAL_15:.*]] = fir.load %[[VAL_13]] {tbaa = [#[[$ATTR_50]]]} : !fir.ref<f32>
// CHECK:           fir.store %[[VAL_15]] to %[[VAL_12]] {tbaa = [#[[$ATTR_48]]]} : !fir.ref<f32>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
func.func @_QMtestPcaller(%arg0: !fir.ref<f32> {fir.bindc_name = "z"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QMtestEx) : !fir.ref<f32>
  %2 = fir.declare %1 {uniq_name = "_QMtestEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %3 = fir.address_of(@_QMtestEy) : !fir.ref<f32>
  %4 = fir.declare %3 {uniq_name = "_QMtestEy"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %5 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QMtestFcallerEz"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %6 = fir.load %4 : !fir.ref<f32>
  fir.store %6 to %2 : !fir.ref<f32>
  %7 = fir.dummy_scope : !fir.dscope
  %8 = fir.address_of(@_QMtestEx) : !fir.ref<f32>
  %9 = fir.declare %8 {uniq_name = "_QMtestEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %10 = fir.address_of(@_QMtestEy) : !fir.ref<f32>
  %11 = fir.declare %10 {uniq_name = "_QMtestEy"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %12 = fir.declare %5 dummy_scope %7 {uniq_name = "_QMtestFcalleeEz"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %13 = fir.load %11 : !fir.ref<f32>
  fir.store %13 to %9 : !fir.ref<f32>
  %14 = fir.load %12 : !fir.ref<f32>
  fir.store %14 to %11 : !fir.ref<f32>
  return
}
fir.global @_QMtestEx : f32 {
  %0 = fir.zero_bits f32
  fir.has_value %0 : f32
}
fir.global @_QMtestEy : f32 {
  %0 = fir.zero_bits f32
  fir.has_value %0 : f32
}
}
