// RUN: fir-opt --fir-add-alias-tags --split-input-file %s | FileCheck %s

// subroutine test(x, y)
//   real, target :: x, y
//   x = y ! the load/store do not have TBAA due to TARGET
//   call inner(x, y) ! the inlined load/store go to Scope 1
//   call inner(x, y) ! the inlined load/store go to Scope 2
// contains
//   subroutine inner(x, y)
//     real :: x, y
//     x = y
//   end subroutine inner
// end subroutine test

// CHECK: #[[$ATTR_0:.+]] = #llvm.tbaa_root<id = "Flang function root test1 - Scope 1">
// CHECK: #[[$ATTR_1:.+]] = #llvm.tbaa_root<id = "Flang function root test1 - Scope 2">
// CHECK: #[[$ATTR_2:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#tbaa_root, 0>}>
// CHECK: #[[$ATTR_3:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#tbaa_root1, 0>}>
// CHECK: #[[$ATTR_4:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#tbaa_type_desc, 0>}>
// CHECK: #[[$ATTR_5:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#tbaa_type_desc1, 0>}>
// CHECK: #[[$ATTR_6:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data", members = {<#tbaa_type_desc2, 0>}>
// CHECK: #[[$ATTR_7:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data", members = {<#tbaa_type_desc3, 0>}>
// CHECK: #[[$ATTR_8:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFtestFinnerEy", members = {<#tbaa_type_desc4, 0>}>
// CHECK: #[[$ATTR_9:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFtestFinnerEx", members = {<#tbaa_type_desc4, 0>}>
// CHECK: #[[$ATTR_10:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFtestFinnerEy", members = {<#tbaa_type_desc5, 0>}>
// CHECK: #[[$ATTR_11:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFtestFinnerEx", members = {<#tbaa_type_desc5, 0>}>
// CHECK: #[[$ATTR_12:.+]] = #llvm.tbaa_tag<base_type = #tbaa_type_desc6, access_type = #tbaa_type_desc6, offset = 0>
// CHECK: #[[$ATTR_13:.+]] = #llvm.tbaa_tag<base_type = #tbaa_type_desc7, access_type = #tbaa_type_desc7, offset = 0>
// CHECK: #[[$ATTR_14:.+]] = #llvm.tbaa_tag<base_type = #tbaa_type_desc8, access_type = #tbaa_type_desc8, offset = 0>
// CHECK: #[[$ATTR_15:.+]] = #llvm.tbaa_tag<base_type = #tbaa_type_desc9, access_type = #tbaa_type_desc9, offset = 0>
// CHECK:   func.func @test1(
// CHECK:           %[[VAL_5:.*]] = fir.load %{{.*}} : !fir.ref<f32>
// CHECK:           fir.store %{{.*}} : !fir.ref<f32>
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_9:.*]] = fir.load %{{.*}} {tbaa = [#[[$ATTR_12]]]} : !fir.ref<f32>
// CHECK:           fir.store %{{.*}} {tbaa = [#[[$ATTR_13]]]} : !fir.ref<f32>
// CHECK:           %[[VAL_10:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_13:.*]] = fir.load %{{.*}} {tbaa = [#[[$ATTR_14]]]} : !fir.ref<f32>
// CHECK:           fir.store %{{.*}} {tbaa = [#[[$ATTR_15]]]} : !fir.ref<f32>
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

// -----

// Check that without proper fir.dummy_scope placement
// we just put everything into the root scope.

// CHECK: #[[$ATTR_16:.+]] = #llvm.tbaa_root<id = "Flang function root test2">
// CHECK: #[[$ATTR_17:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#tbaa_root, 0>}>
// CHECK: #[[$ATTR_18:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#tbaa_type_desc, 0>}>
// CHECK: #[[$ATTR_19:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data", members = {<#tbaa_type_desc1, 0>}>
// CHECK: #[[$ATTR_20:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFtestEy", members = {<#tbaa_type_desc2, 0>}>
// CHECK: #[[$ATTR_21:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QFtestEx", members = {<#tbaa_type_desc2, 0>}>
// CHECK: #[[$ATTR_22:.+]] = #llvm.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc3, offset = 0>
// CHECK: #[[$ATTR_23:.+]] = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc4, offset = 0>
// CHECK:   func.func @test2(
// CHECK:           %[[VAL_4:.*]] = fir.load %{{.*}} {tbaa = [#[[$ATTR_22]]]} : !fir.ref<f32>
// CHECK:           fir.store %{{.*}} {tbaa = [#[[$ATTR_23]]]} : !fir.ref<f32>
// CHECK:           %[[VAL_5:.*]] = fir.declare
// CHECK:           %[[VAL_6:.*]] = fir.declare
// CHECK:           %[[VAL_7:.*]] = fir.load %{{.*}} {tbaa = [#[[$ATTR_22]]]} : !fir.ref<f32>
// CHECK:           fir.store %{{.*}} {tbaa = [#[[$ATTR_23]]]} : !fir.ref<f32>
func.func @test2(%arg0: !fir.ref<f32> {fir.bindc_name = "x"}, %arg1: !fir.ref<f32> {fir.bindc_name = "y"}) {
  %0 = fir.declare %arg0 {uniq_name = "_QFtestEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %1 = fir.declare %arg1 {uniq_name = "_QFtestEy"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %2 = fir.load %1 : !fir.ref<f32>
  fir.store %2 to %0 : !fir.ref<f32>
  %3 = fir.declare %0 {uniq_name = "_QFtestFinnerEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %4 = fir.declare %1 {uniq_name = "_QFtestFinnerEy"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %5 = fir.load %4 : !fir.ref<f32>
  fir.store %5 to %3 : !fir.ref<f32>
  return
}

// -----

// module test
//   real :: x, y
// contains
// subroutine caller
//   x = y ! the load/store go to the root scope
//   call callee
// end subroutine caller
// subroutine callee
//   x = y ! the load/store go to the root scope
//   ! Since there are no dummy arguments in callee,
//   ! it is better to put the load/store into the
//   ! enclosing root scope, so that they can be
//   ! disambiguated using TBAA with the loads/stores
//   ! in the enclosing scope.
// end subroutine callee
// end module test

// CHECK: #[[$ATTR_24:.+]] = #llvm.tbaa_root<id = "Flang function root _QMtestPcaller">
// CHECK: #[[$ATTR_25:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#tbaa_root, 0>}>
// CHECK: #[[$ATTR_26:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#tbaa_type_desc, 0>}>
// CHECK: #[[$ATTR_27:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#tbaa_type_desc1, 0>}>
// CHECK: #[[$ATTR_28:.+]] = #llvm.tbaa_type_desc<id = "global data/_QMtestEy", members = {<#tbaa_type_desc2, 0>}>
// CHECK: #[[$ATTR_29:.+]] = #llvm.tbaa_type_desc<id = "global data/_QMtestEx", members = {<#tbaa_type_desc2, 0>}>
// CHECK: #[[$ATTR_30:.+]] = #llvm.tbaa_tag<base_type = #tbaa_type_desc3, access_type = #tbaa_type_desc3, offset = 0>
// CHECK: #[[$ATTR_31:.+]] = #llvm.tbaa_tag<base_type = #tbaa_type_desc4, access_type = #tbaa_type_desc4, offset = 0>
// CHECK:   func.func @_QMtestPcaller() {
// CHECK:           %[[VAL_0:.*]] = fir.address_of(@_QMtestEx) : !fir.ref<f32>
// CHECK:           %[[VAL_1:.*]] = fir.declare %[[VAL_0]] {uniq_name = "_QMtestEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
// CHECK:           %[[VAL_2:.*]] = fir.address_of(@_QMtestEy) : !fir.ref<f32>
// CHECK:           %[[VAL_3:.*]] = fir.declare %[[VAL_2]] {uniq_name = "_QMtestEy"} : (!fir.ref<f32>) -> !fir.ref<f32>
// CHECK:           %[[VAL_4:.*]] = fir.load %[[VAL_3]] {tbaa = [#[[$ATTR_30]]]} : !fir.ref<f32>
// CHECK:           fir.store %[[VAL_4]] to %[[VAL_1]] {tbaa = [#[[$ATTR_31]]]} : !fir.ref<f32>
// CHECK:           %[[VAL_5:.*]] = fir.address_of(@_QMtestEx) : !fir.ref<f32>
// CHECK:           %[[VAL_6:.*]] = fir.declare %[[VAL_5]] {uniq_name = "_QMtestEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
// CHECK:           %[[VAL_7:.*]] = fir.address_of(@_QMtestEy) : !fir.ref<f32>
// CHECK:           %[[VAL_8:.*]] = fir.declare %[[VAL_7]] {uniq_name = "_QMtestEy"} : (!fir.ref<f32>) -> !fir.ref<f32>
// CHECK:           %[[VAL_9:.*]] = fir.load %[[VAL_8]] {tbaa = [#[[$ATTR_30]]]} : !fir.ref<f32>
// CHECK:           fir.store %[[VAL_9]] to %[[VAL_6]] {tbaa = [#[[$ATTR_31]]]} : !fir.ref<f32>
func.func @_QMtestPcaller() {
  %0 = fir.address_of(@_QMtestEx) : !fir.ref<f32>
  %1 = fir.declare %0 {uniq_name = "_QMtestEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %2 = fir.address_of(@_QMtestEy) : !fir.ref<f32>
  %3 = fir.declare %2 {uniq_name = "_QMtestEy"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %4 = fir.load %3 : !fir.ref<f32>
  fir.store %4 to %1 : !fir.ref<f32>
  %5 = fir.address_of(@_QMtestEx) : !fir.ref<f32>
  %6 = fir.declare %5 {uniq_name = "_QMtestEx"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %7 = fir.address_of(@_QMtestEy) : !fir.ref<f32>
  %8 = fir.declare %7 {uniq_name = "_QMtestEy"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %9 = fir.load %8 : !fir.ref<f32>
  fir.store %9 to %6 : !fir.ref<f32>
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
// CHECK: #[[$ATTR_34:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#tbaa_root, 0>}>
// CHECK: #[[$ATTR_35:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#tbaa_root1, 0>}>
// CHECK: #[[$ATTR_36:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#tbaa_type_desc, 0>}>
// CHECK: #[[$ATTR_37:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#tbaa_type_desc1, 0>}>
// CHECK: #[[$ATTR_38:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#tbaa_type_desc2, 0>}>
// CHECK: #[[$ATTR_39:.+]] = #llvm.tbaa_type_desc<id = "global data", members = {<#tbaa_type_desc3, 0>}>
// CHECK: #[[$ATTR_40:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data", members = {<#tbaa_type_desc3, 0>}>
// CHECK: #[[$ATTR_41:.+]] = #llvm.tbaa_type_desc<id = "global data/_QMtestEy", members = {<#tbaa_type_desc4, 0>}>
// CHECK: #[[$ATTR_42:.+]] = #llvm.tbaa_type_desc<id = "global data/_QMtestEx", members = {<#tbaa_type_desc4, 0>}>
// CHECK: #[[$ATTR_43:.+]] = #llvm.tbaa_type_desc<id = "global data/_QMtestEy", members = {<#tbaa_type_desc5, 0>}>
// CHECK: #[[$ATTR_44:.+]] = #llvm.tbaa_type_desc<id = "global data/_QMtestEx", members = {<#tbaa_type_desc5, 0>}>
// CHECK: #[[$ATTR_45:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QMtestFcalleeEz", members = {<#tbaa_type_desc6, 0>}>
// CHECK: #[[$ATTR_46:.+]] = #llvm.tbaa_tag<base_type = #tbaa_type_desc7, access_type = #tbaa_type_desc7, offset = 0>
// CHECK: #[[$ATTR_47:.+]] = #llvm.tbaa_tag<base_type = #tbaa_type_desc8, access_type = #tbaa_type_desc8, offset = 0>
// CHECK: #[[$ATTR_48:.+]] = #llvm.tbaa_tag<base_type = #tbaa_type_desc9, access_type = #tbaa_type_desc9, offset = 0>
// CHECK: #[[$ATTR_49:.+]] = #llvm.tbaa_tag<base_type = #tbaa_type_desc10, access_type = #tbaa_type_desc10, offset = 0>
// CHECK: #[[$ATTR_50:.+]] = #llvm.tbaa_tag<base_type = #tbaa_type_desc11, access_type = #tbaa_type_desc11, offset = 0>
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
