! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Sections of named constants as actual arguments: a dynamic section keeps
! its descriptor (strides intact) for an assumed-shape dummy, gets a
! temporary copy for an explicit-shape dummy, and a temporary made from a
! named constant is never copied back into the constant's storage.

module ms
  implicit none
  integer, parameter :: sp(6) = [1, 2, 3, 4, 5, 6]
contains
  subroutine expl3(x)
    integer, intent(in) :: x(3)
  end subroutine
  subroutine asmd(x)
    integer, intent(in) :: x(:)
  end subroutine
end module

! Dynamic-bound strided section to an assumed-shape dummy: the descriptor
! over the named constant's storage is passed directly, strides preserved;
! no copy in either direction.
! CHECK-LABEL: func.func @_QPdyn_section_assumed
! CHECK: %[[A:.*]] = fir.address_of(@_QMmsECsp)
! CHECK: %[[AD:.*]]:2 = hlfir.declare %[[A]]
! CHECK: %[[SEC:.*]] = hlfir.designate %[[AD]]#0 (%{{.*}}:%c6{{.*}}:%c2{{.*}}) shape %{{.*}} -> !fir.box<!fir.array<?xi32>>
! CHECK-NOT: hlfir.as_expr
! CHECK-NOT: hlfir.copy_in
! CHECK: fir.call @_QMmsPasmd(%[[SEC]])
! CHECK-NOT: hlfir.copy_out
subroutine dyn_section_assumed(i)
  use ms
  integer :: i
  call asmd(sp(i:6:2))
end subroutine

! The same section to an explicit-shape dummy requires contiguity: a
! temporary copy is made, and nothing is copied back into the named
! constant's storage afterwards (the temporary is simply destroyed).
! CHECK-LABEL: func.func @_QPdyn_section_explicit
! CHECK: %[[B:.*]] = fir.address_of(@_QMmsECsp)
! CHECK: %[[BD:.*]]:2 = hlfir.declare %[[B]]
! CHECK: %[[BSEC:.*]] = hlfir.designate %[[BD]]#0 (%{{.*}}:%c6{{.*}}:%c2{{.*}}) shape
! CHECK: %[[BEXPR:.*]] = hlfir.as_expr %[[BSEC]]
! CHECK: %[[BTMP:.*]]:3 = hlfir.associate %[[BEXPR]]({{.*}}) {adapt.valuebyref}
! CHECK: fir.call @_QMmsPexpl3
! CHECK-NOT: hlfir.copy_out
! CHECK: hlfir.end_associate %[[BTMP]]#1, %[[BTMP]]#2
subroutine dyn_section_explicit(i)
  use ms
  integer :: i
  call expl3(sp(i:6:2))
end subroutine

! A section with constant bounds is folded to an outlined constant, which
! is copied to a temporary; still no copy-out.
! CHECK-LABEL: func.func @_QPconst_strided_explicit
! CHECK: fir.address_of(@_QQro.3xi4.0)
! CHECK: hlfir.as_expr
! CHECK: %[[CTMP:.*]]:3 = hlfir.associate {{.*}} {adapt.valuebyref}
! CHECK: fir.call @_QMmsPexpl3(%[[CTMP]]#0)
! CHECK-NOT: hlfir.copy_out
! CHECK: hlfir.end_associate
subroutine const_strided_explicit()
  use ms
  call expl3(sp(1:6:2))
end subroutine
