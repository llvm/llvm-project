! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Defined assignment with a named-constant right-hand side.  Assignment
! operands are expressions, not procedure-reference arguments, so the named
! constant arrives folded (an outlined constant) and the defined assignment
! is lowered through hlfir.region_assign; the copy analysis that drives the
! parameter-object temporary in ordinary calls sees nothing to copy here, and
! no temporary is made inside the user_defined_assign region.

module mda
  implicit none
  type t
    integer :: val
  end type
  type(t), parameter :: cp = t(3)
  type(t), parameter :: cpa(2) = [t(1), t(2)]
  interface assignment(=)
    module procedure custom_assign
    module procedure custom_assign_array
  end interface
contains
  subroutine custom_assign(lhs, rhs)
    type(t), intent(out) :: lhs
    type(t), intent(in) :: rhs
    lhs%val = rhs%val * 10
  end subroutine
  subroutine custom_assign_array(lhs, rhs)
    type(t), intent(out) :: lhs(:)
    type(t), intent(in) :: rhs(:)
    lhs%val = rhs%val * 10
  end subroutine
end module

! CHECK-LABEL: func.func @_QPdefined_assign_scalar_parameter
! CHECK: hlfir.region_assign {
! CHECK: %[[RHS:.*]] = fir.address_of(@_QQro._QMmdaTt.0)
! CHECK: %[[RHSD:.*]]:2 = hlfir.declare %[[RHS]] {fortran_attrs = #fir.var_attrs<parameter>
! CHECK: hlfir.yield %[[RHSD]]#0
! CHECK: } to {
! CHECK: } user_defined_assign (%[[ARG0:.*]]: !fir.ref<!fir.type<_QMmdaTt{val:i32}>>) to (%[[ARG1:.*]]: !fir.ref<!fir.type<_QMmdaTt{val:i32}>>) {
! CHECK-NOT: hlfir.as_expr
! CHECK-NOT: hlfir.associate
! CHECK: fir.call @_QMmdaPcustom_assign(%[[ARG1]], %[[ARG0]])
subroutine defined_assign_scalar_parameter()
  use mda
  type(t) :: x
  x = cp
end subroutine

! CHECK-LABEL: func.func @_QPdefined_assign_array_parameter
! CHECK: hlfir.region_assign {
! CHECK: %[[ARHS:.*]] = fir.address_of(@_QQro.2x_QMmdaTt.1)
! CHECK: %[[ARHSD:.*]]:2 = hlfir.declare %[[ARHS]](%{{.*}}) {fortran_attrs = #fir.var_attrs<parameter>
! CHECK: hlfir.yield %[[ARHSD]]#0
! CHECK: } to {
! CHECK: } user_defined_assign (%[[AARG0:.*]]: !fir.ref<!fir.array<2x!fir.type<_QMmdaTt{val:i32}>>>) to (%[[AARG1:.*]]: !fir.ref<!fir.array<2x!fir.type<_QMmdaTt{val:i32}>>>) {
! CHECK-NOT: hlfir.as_expr
! CHECK-NOT: hlfir.associate
! CHECK: %[[LHSBOX:.*]] = fir.embox %[[AARG1]]
! CHECK: %[[LHS:.*]] = fir.convert %[[LHSBOX]]
! CHECK: %[[RHSBOX:.*]] = fir.embox %[[AARG0]]
! CHECK: %[[RHS2:.*]] = fir.convert %[[RHSBOX]]
! CHECK-NOT: hlfir.as_expr
! CHECK-NOT: hlfir.associate
! CHECK: fir.call @_QMmdaPcustom_assign_array(%[[LHS]], %[[RHS2]])
subroutine defined_assign_array_parameter()
  use mda
  type(t) :: y(2)
  y = cpa
end subroutine
