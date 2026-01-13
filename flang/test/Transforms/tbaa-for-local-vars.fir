// Test TBAA tags assignment for access off the fir.declare
// placed in the middle of the routine (e.g. a temporary).
// Original example:
// module m
//   type t
//      real :: x
//   end type t
// contains
//   subroutine bar(this)
//     class(t), intent(out) :: this
//     this%x = 1.0
//   end subroutine bar
//   function foo() result(res)
//     type(t) :: res
//     call bar(res)
//   end function foo
//   subroutine test(arg)
//     type(t) :: var
//     var = foo()
//     arg = var%x
//   end subroutine test
// end module m
//
// The calls were manually inlined in FIR with fir.save_result's
// destination operand being used instead of the temporary
// alloca of the result inside foo. Runtime calls were removed
// to reduce the test.
// The temporary function result fir.declare is then dominated
// by fir.dummy_scope of the foo function. If we use this scope
// to assign the TBAA local-alloc tag to the accesses of
// the temporary, then it won't conflict with the TBAA dummy tag
// assigned to the accesses of this argument of bar.
// That would be incorrect, because they access the same memory.
// Check that the local-alloc tag is placed into the outermost
// scope's TBAA tree.
// RUN: fir-opt --fir-add-alias-tags %s | FileCheck %s

// CHECK: #[[$SCOPE_2:.+]] = #llvm.tbaa_root<id = "Flang function root _QMmPtest - Scope 2">
// CHECK: #[[$SCOPE_1:.+]] = #llvm.tbaa_root<id = "Flang function root _QMmPtest">
// CHECK: #[[$ANY_ACCESS2:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[$SCOPE_2]], 0>}>
// CHECK: #[[$ANY_ACCESS1:.+]] = #llvm.tbaa_type_desc<id = "any access", members = {<#[[$SCOPE_1]], 0>}>
// CHECK: #[[$ANY_DATA2:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[$ANY_ACCESS2]], 0>}>
// CHECK: #[[$ANY_DATA1:.+]] = #llvm.tbaa_type_desc<id = "any data access", members = {<#[[$ANY_ACCESS1]], 0>}>
// CHECK: #[[$DUMMY_ARG2:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data", members = {<#[[$ANY_DATA2]], 0>}>
// CHECK: #[[$ALLOCATED_DATA1:.+]] = #llvm.tbaa_type_desc<id = "allocated data", members = {<#[[$ANY_DATA1]], 0>}>
// CHECK: #[[$DUMMY_ARG1:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data", members = {<#[[$ANY_DATA1]], 0>}>
// CHECK: #[[$ALLOCATED_DATA1_TAG:.+]] = #llvm.tbaa_tag<base_type = #[[$ALLOCATED_DATA1]], access_type = #[[$ALLOCATED_DATA1]], offset = 0>
// CHECK: #[[$BAR_THIS2:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QMmFbarEthis", members = {<#[[$DUMMY_ARG2]], 0>}>
// CHECK: #[[$TEST_VAR1:.+]] = #llvm.tbaa_type_desc<id = "allocated data/_QMmFtestEvar", members = {<#[[$ALLOCATED_DATA1]], 0>}>
// CHECK: #[[$TEST_ARG1:.+]] = #llvm.tbaa_type_desc<id = "dummy arg data/_QMmFtestEarg", members = {<#[[$DUMMY_ARG1]], 0>}>
// CHECK: #[[$BAR_THIS2_TAG:.+]] = #llvm.tbaa_tag<base_type = #[[$BAR_THIS2]], access_type = #[[$BAR_THIS2]], offset = 0>
// CHECK: #[[$TEST_VAR1_TAG:.+]] = #llvm.tbaa_tag<base_type = #[[$TEST_VAR1]], access_type = #[[$TEST_VAR1]], offset = 0>
// CHECK: #[[$TEST_ARG2_TAG:.+]] = #llvm.tbaa_tag<base_type = #[[$TEST_ARG1]], access_type = #[[$TEST_ARG1]], offset = 0>

// CHECK-LABEL:   func.func @_QMmPtest(
// CHECK-SAME:      %[[ARG0:.*]]: !fir.ref<f32> {fir.bindc_name = "arg"}) {
// CHECK:           %[[VAL_0:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:           %[[VAL_1:.*]] = fir.alloca !fir.type<_QMmTt{x:f32}> {bindc_name = ".result"}
// CHECK:           %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_3:.*]] = fir.declare %[[ARG0]] dummy_scope %[[VAL_2]] {uniq_name = "_QMmFtestEarg"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
// CHECK:           %[[VAL_6:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_7:.*]] = fir.declare %[[VAL_1]] {uniq_name = "_QMmFfooEres"} : (!fir.ref<!fir.type<_QMmTt{x:f32}>>) -> !fir.ref<!fir.type<_QMmTt{x:f32}>>
// CHECK:           %[[VAL_8:.*]] = fir.embox %[[VAL_7]] : (!fir.ref<!fir.type<_QMmTt{x:f32}>>) -> !fir.box<!fir.type<_QMmTt{x:f32}>>
// CHECK:           %[[VAL_9:.*]] = fir.convert %[[VAL_8]] : (!fir.box<!fir.type<_QMmTt{x:f32}>>) -> !fir.class<!fir.type<_QMmTt{x:f32}>>
// CHECK:           %[[VAL_10:.*]] = fir.dummy_scope : !fir.dscope
// CHECK:           %[[VAL_11:.*]] = fir.declare %[[VAL_9]] dummy_scope %[[VAL_10]] {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "_QMmFbarEthis"} : (!fir.class<!fir.type<_QMmTt{x:f32}>>, !fir.dscope) -> !fir.class<!fir.type<_QMmTt{x:f32}>>
// CHECK:           %[[VAL_12:.*]] = fir.coordinate_of %[[VAL_11]], x : (!fir.class<!fir.type<_QMmTt{x:f32}>>) -> !fir.ref<f32>
// CHECK:           fir.store %[[VAL_0]] to %[[VAL_12]] {tbaa = [#[[$BAR_THIS2_TAG]]]} : !fir.ref<f32>
// CHECK:           %[[VAL_13:.*]] = fir.declare %[[VAL_1]] {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.type<_QMmTt{x:f32}>>) -> !fir.ref<!fir.type<_QMmTt{x:f32}>>
// CHECK:           %[[VAL_14:.*]] = fir.coordinate_of %[[VAL_13]], x : (!fir.ref<!fir.type<_QMmTt{x:f32}>>) -> !fir.ref<f32>
// CHECK:           %[[VAL_16:.*]] = fir.load %[[VAL_14]] {tbaa = [#[[$ALLOCATED_DATA1_TAG]]]} : !fir.ref<f32>
module attributes {dlti.dl_spec = #dlti.dl_spec<!llvm.ptr = dense<64> : vector<4xi64>, i1 = dense<8> : vector<2xi64>, i8 = dense<8> : vector<2xi64>, i16 = dense<16> : vector<2xi64>, i32 = dense<32> : vector<2xi64>, i64 = dense<[32, 64]> : vector<2xi64>, f16 = dense<16> : vector<2xi64>, f64 = dense<64> : vector<2xi64>, f128 = dense<128> : vector<2xi64>, "dlti.endianness" = "little">, llvm.data_layout = ""} {
func.func @_QMmPtest(%arg0: !fir.ref<f32> {fir.bindc_name = "arg"}) {
  %cst = arith.constant 1.000000e+00 : f32
  %0 = fir.alloca !fir.type<_QMmTt{x:f32}> {bindc_name = ".result"}
  %1 = fir.dummy_scope : !fir.dscope
  %2 = fir.declare %arg0 dummy_scope %1 {uniq_name = "_QMmFtestEarg"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %3 = fir.alloca !fir.type<_QMmTt{x:f32}> {bindc_name = "var", uniq_name = "_QMmFtestEvar"}
  %4 = fir.declare %3 {uniq_name = "_QMmFtestEvar"} : (!fir.ref<!fir.type<_QMmTt{x:f32}>>) -> !fir.ref<!fir.type<_QMmTt{x:f32}>>
  %5 = fir.dummy_scope : !fir.dscope
  %6 = fir.declare %0 {uniq_name = "_QMmFfooEres"} : (!fir.ref<!fir.type<_QMmTt{x:f32}>>) -> !fir.ref<!fir.type<_QMmTt{x:f32}>>
  %7 = fir.embox %6 : (!fir.ref<!fir.type<_QMmTt{x:f32}>>) -> !fir.box<!fir.type<_QMmTt{x:f32}>>
  %8 = fir.convert %7 : (!fir.box<!fir.type<_QMmTt{x:f32}>>) -> !fir.class<!fir.type<_QMmTt{x:f32}>>
  %9 = fir.dummy_scope : !fir.dscope
  %10 = fir.declare %8 dummy_scope %9 {fortran_attrs = #fir.var_attrs<intent_out>, uniq_name = "_QMmFbarEthis"} : (!fir.class<!fir.type<_QMmTt{x:f32}>>, !fir.dscope) -> !fir.class<!fir.type<_QMmTt{x:f32}>>
  %14 = fir.coordinate_of %10, x : (!fir.class<!fir.type<_QMmTt{x:f32}>>) -> !fir.ref<f32>
  fir.store %cst to %14 : !fir.ref<f32>
  %15 = fir.declare %0 {uniq_name = ".tmp.func_result"} : (!fir.ref<!fir.type<_QMmTt{x:f32}>>) -> !fir.ref<!fir.type<_QMmTt{x:f32}>>
  %16 = fir.coordinate_of %15, x : (!fir.ref<!fir.type<_QMmTt{x:f32}>>) -> !fir.ref<f32>
  %17 = fir.coordinate_of %4, x : (!fir.ref<!fir.type<_QMmTt{x:f32}>>) -> !fir.ref<f32>
  %18 = fir.load %16 : !fir.ref<f32>
  fir.store %18 to %17 : !fir.ref<f32>
  %19 = fir.load %17 : !fir.ref<f32>
  fir.store %19 to %2 : !fir.ref<f32>
  return
}
}
