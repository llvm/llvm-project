// RUN: fir-opt --opt-bufferization %s | FileCheck %s

// tonto case where optimized bufferization conservatively
// did not pull elemental into the assignment due to
// the designate op in between:
// module test
//   type my_type
//      real, dimension(:,:), pointer :: p => null()
//   end type my_type
// contains
//   subroutine assign1(self, i)
//     type(my_type) :: self
//     real, dimension(3) :: ct, q
//     integer :: i
//     self%p(:,i) = q - ct
//   end subroutine assign1
// end module test

// CHECK-LABEL:   func.func @_QMtestPassign1(
// CHECK-NOT: hlfir.elemental
// CHECK-NOT: hlfir.assign{{.*}}array
func.func @_QMtestPassign1(%arg0: !fir.ref<!fir.type<_QMtestTmy_type{p:!fir.box<!fir.ptr<!fir.array<?x?xf32>>>}>> {fir.bindc_name = "self"}, %arg1: !fir.ref<i32> {fir.bindc_name = "i"}) {
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  %c3 = arith.constant 3 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca !fir.array<3xf32> {bindc_name = "ct", uniq_name = "_QMtestFassign1Ect"}
  %2 = fir.shape %c3 : (index) -> !fir.shape<1>
  %3:2 = hlfir.declare %1(%2) {uniq_name = "_QMtestFassign1Ect"} : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xf32>>, !fir.ref<!fir.array<3xf32>>)
  %4:2 = hlfir.declare %arg1 dummy_scope %0 {uniq_name = "_QMtestFassign1Ei"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
  %5 = fir.alloca !fir.array<3xf32> {bindc_name = "q", uniq_name = "_QMtestFassign1Eq"}
  %6:2 = hlfir.declare %5(%2) {uniq_name = "_QMtestFassign1Eq"} : (!fir.ref<!fir.array<3xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<3xf32>>, !fir.ref<!fir.array<3xf32>>)
  %7:2 = hlfir.declare %arg0 dummy_scope %0 {uniq_name = "_QMtestFassign1Eself"} : (!fir.ref<!fir.type<_QMtestTmy_type{p:!fir.box<!fir.ptr<!fir.array<?x?xf32>>>}>>, !fir.dscope) -> (!fir.ref<!fir.type<_QMtestTmy_type{p:!fir.box<!fir.ptr<!fir.array<?x?xf32>>>}>>, !fir.ref<!fir.type<_QMtestTmy_type{p:!fir.box<!fir.ptr<!fir.array<?x?xf32>>>}>>)
  %8 = hlfir.elemental %2 unordered : (!fir.shape<1>) -> !hlfir.expr<3xf32> {
  ^bb0(%arg2: index):
    %22 = hlfir.designate %6#0 (%arg2)  : (!fir.ref<!fir.array<3xf32>>, index) -> !fir.ref<f32>
    %23 = hlfir.designate %3#0 (%arg2)  : (!fir.ref<!fir.array<3xf32>>, index) -> !fir.ref<f32>
    %24 = fir.load %22 : !fir.ref<f32>
    %25 = fir.load %23 : !fir.ref<f32>
    %26 = arith.subf %24, %25 fastmath<contract> : f32
    hlfir.yield_element %26 : f32
  }
  %9 = hlfir.designate %7#0{"p"}   {fortran_attrs = #fir.var_attrs<pointer>} : (!fir.ref<!fir.type<_QMtestTmy_type{p:!fir.box<!fir.ptr<!fir.array<?x?xf32>>>}>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  %10 = fir.load %9 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?x?xf32>>>>
  %11:3 = fir.box_dims %10, %c0 : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index) -> (index, index, index)
  %12 = arith.addi %11#0, %11#1 : index
  %13 = arith.subi %12, %c1 : index
  %14 = arith.subi %13, %11#0 : index
  %15 = arith.addi %14, %c1 : index
  %16 = arith.cmpi sgt, %15, %c0 : index
  %17 = arith.select %16, %15, %c0 : index
  %18 = fir.load %4#0 : !fir.ref<i32>
  %19 = fir.convert %18 : (i32) -> i64
  %20 = fir.shape %17 : (index) -> !fir.shape<1>
  %21 = hlfir.designate %10 (%11#0:%13:%c1, %19)  shape %20 : (!fir.box<!fir.ptr<!fir.array<?x?xf32>>>, index, index, index, i64, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
  hlfir.assign %8 to %21 : !hlfir.expr<3xf32>, !fir.box<!fir.array<?xf32>>
  hlfir.destroy %8 : !hlfir.expr<3xf32>
  return
}
