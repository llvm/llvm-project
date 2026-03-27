// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))'  2>&1 | FileCheck %s

// The test was obtained from 
// bbc test.f90 -emit-fir
// module mod
//   real, pointer :: p0
//   real, allocatable :: alloc
//   real, allocatable, target :: t_alloc
//   real, target :: t
//   real :: v
// end module
// 
// subroutine test(n)
//   use mod
//   integer :: n
//   real r1
//   p0 => t_alloc
//   v = alloc
//   r1 = p0
// end subroutine test

// Checking that aliasing can only happen with an entity with the target attribute
// 
// CHECK-DAG: r1#0 <-> t_alloc#0: NoAlias
// CHECK-DAG: r1#0 <-> alloc#0: NoAlias
// CHECK-DAG: t_alloc#0 <-> alloc#0: NoAlias
// CHECK-DAG: r1#0 <-> p0.ptr#0: NoAlias
// CHECK-DAG: t_alloc#0 <-> p0.ptr#0: MayAlias
// CHECK-DAG: alloc#0 <-> p0.ptr#0: NoAlias

fir.global @_QMmodEalloc : !fir.box<!fir.heap<f32>> {
  %0 = fir.zero_bits !fir.heap<f32>
  %1 = fir.embox %0 : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
  fir.has_value %1 : !fir.box<!fir.heap<f32>>
}
fir.global @_QMmodEp0 : !fir.box<!fir.ptr<f32>> {
  %0 = fir.zero_bits !fir.ptr<f32>
  %1 = fir.embox %0 : (!fir.ptr<f32>) -> !fir.box<!fir.ptr<f32>>
  fir.has_value %1 : !fir.box<!fir.ptr<f32>>
}
fir.global @_QMmodEt target : f32 {
  %0 = fir.zero_bits f32
  fir.has_value %0 : f32
}
fir.global @_QMmodEt_alloc target : !fir.box<!fir.heap<f32>> {
  %0 = fir.zero_bits !fir.heap<f32>
  %1 = fir.embox %0 : (!fir.heap<f32>) -> !fir.box<!fir.heap<f32>>
  fir.has_value %1 : !fir.box<!fir.heap<f32>>
}
fir.global @_QMmodEv : f32 {
  %0 = fir.zero_bits f32
  fir.has_value %0 : f32
}
func.func @_QPtest(%arg0: !fir.ref<i32> {fir.bindc_name = "n"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.address_of(@_QMmodEalloc) : !fir.ref<!fir.box<!fir.heap<f32>>>
  %2 = fir.declare %1 {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QMmodEalloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %3 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QFtestEn"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %4 = fir.address_of(@_QMmodEp0) : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %5 = fir.declare %4 {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QMmodEp0"} : (!fir.ref<!fir.box<!fir.ptr<f32>>>) -> !fir.ref<!fir.box<!fir.ptr<f32>>>
  %6 = fir.alloca f32 {bindc_name = "r1", uniq_name = "_QFtestEr1"}
  %7 = fir.declare %6 {test.ptr="r1", uniq_name = "_QFtestEr1"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %8 = fir.address_of(@_QMmodEt) : !fir.ref<f32>
  %9 = fir.declare %8 {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QMmodEt"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %10 = fir.address_of(@_QMmodEt_alloc) : !fir.ref<!fir.box<!fir.heap<f32>>>
  %11 = fir.declare %10 {fortran_attrs = #fir.var_attrs<allocatable, target>, uniq_name = "_QMmodEt_alloc"} : (!fir.ref<!fir.box<!fir.heap<f32>>>) -> !fir.ref<!fir.box<!fir.heap<f32>>>
  %12 = fir.address_of(@_QMmodEv) : !fir.ref<f32>
  %13 = fir.declare %12 {uniq_name = "_QMmodEv"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %14 = fir.load %11 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %15 = fir.box_addr %14 {test.ptr="t_alloc"}: (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %16 = fir.embox %15 : (!fir.heap<f32>) -> !fir.box<!fir.ptr<f32>>
  fir.store %16 to %5 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %17 = fir.load %2 : !fir.ref<!fir.box<!fir.heap<f32>>>
  %18 = fir.box_addr %17 {test.ptr="alloc"} : (!fir.box<!fir.heap<f32>>) -> !fir.heap<f32>
  %19 = fir.load %18 : !fir.heap<f32>
  fir.store %19 to %13 : !fir.ref<f32>
  %20 = fir.load %5 : !fir.ref<!fir.box<!fir.ptr<f32>>>
  %21 = fir.box_addr %20 {test.ptr="p0.ptr"} : (!fir.box<!fir.ptr<f32>>) -> !fir.ptr<f32>
  %22 = fir.load %21 : !fir.ptr<f32>
  fir.store %22 to %7 : !fir.ref<f32>
  return
}
