! Test lowering of elemental calls in array expressions.
! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

module scalar_in_elem

contains
elemental integer function elem_by_ref(a,b) result(r)
  integer, intent(in) :: a
  real, intent(in) :: b
  r = a + b
end function
elemental integer function elem_by_valueref(a,b) result(r)
  integer, value :: a
  real, value :: b
  r = a + b
end function

! CHECK-LABEL: func @_QMscalar_in_elemPtest_elem_by_ref(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.array<100xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<!fir.array<100xi32>>{{.*}}) {
subroutine test_elem_by_ref(i, j)
  integer :: i(100), j(100)
  ! CHECK: %[[i_decl:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK: %[[j_decl:.*]]:2 = hlfir.declare %[[arg1]]
  ! CHECK: %[[cst:.*]] = arith.constant 4.200000e+01 : f32
  ! CHECK: hlfir.elemental {{.*}} {
  ! CHECK: ^bb0(%[[idx:.*]]: index):
  ! CHECK:   %[[j_elem:.*]] = hlfir.designate %[[j_decl]]#0 (%[[idx]])
  ! CHECK:   %[[assoc:.*]]:3 = hlfir.associate %[[cst]]
  ! CHECK:   fir.call @_QMscalar_in_elemPelem_by_ref(%[[j_elem]], %[[assoc]]#0)
  ! CHECK: }
  i = elem_by_ref(j, 42.)
end

! CHECK-LABEL: func @_QMscalar_in_elemPtest_elem_by_valueref(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<!fir.array<100xi32>>{{.*}}, %[[arg1:.*]]: !fir.ref<!fir.array<100xi32>>{{.*}}) {
subroutine test_elem_by_valueref(i, j)
  integer :: i(100), j(100)
  ! CHECK: %[[i_decl:.*]]:2 = hlfir.declare %[[arg0]]
  ! CHECK: %[[j_decl:.*]]:2 = hlfir.declare %[[arg1]]
  ! CHECK: %[[cst:.*]] = arith.constant 4.200000e+01 : f32
  ! CHECK: hlfir.elemental {{.*}} {
  ! CHECK: ^bb0(%[[idx:.*]]: index):
  ! CHECK:   %[[j_elem_addr:.*]] = hlfir.designate %[[j_decl]]#0 (%[[idx]])
  ! CHECK:   %[[j_val:.*]] = fir.load %[[j_elem_addr]]
  ! CHECK:   fir.call @_QMscalar_in_elemPelem_by_valueref(%[[j_val]], %[[cst]])
  ! CHECK: }
  i = elem_by_valueref(j, 42.)
end
end module


! Test that impure elemental functions cause ordered loops to be emitted
subroutine test_loop_order(i, j)
  integer :: i(:), j(:)
  interface
    elemental integer function pure_func(j)
      integer, intent(in) :: j
    end function
    elemental impure integer function impure_func(j)
      integer, intent(in) :: j
    end function
  end interface

  i = 42 + pure_func(j)
  i = 42 + impure_func(j)
end subroutine

! CHECK-LABEL: func @_QPtest_loop_order(
! CHECK-SAME:    %[[i:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}, %[[j:.*]]: !fir.box<!fir.array<?xi32>>{{.*}}) {
! CHECK:         %[[i_decl:.*]]:2 = hlfir.declare %[[i]]
! CHECK:         %[[j_decl:.*]]:2 = hlfir.declare %[[j]]
! CHECK:         hlfir.elemental {{.*}} unordered
! CHECK:           fir.call @_QPpure_func
! CHECK:         hlfir.assign

! CHECK:         hlfir.elemental {{.*}} : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
! CHECK-NOT:     unordered
! CHECK:           fir.call @_QPimpure_func
! CHECK:         hlfir.assign
! CHECK:       }
