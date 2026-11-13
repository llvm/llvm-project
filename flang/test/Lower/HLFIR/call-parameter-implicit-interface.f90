! RUN: %flang_fc1 -emit-hlfir %s -o - 2>/dev/null | FileCheck %s

! Test that named-constant (PARAMETER) arrays and sections passed through an
! implicit interface are copied into a temporary, since the procedure may
! define the dummy argument, while array elements keep the named constant's
! storage (they may start a sequence association) and calls through an
! explicit interface still pass the storage directly.

module m
  implicit none
contains
  subroutine expl(x)
    integer :: x(4)
  end subroutine
end module

! Whole named-constant array to an external without an interface.
! CHECK-LABEL: func.func @_QPwhole_implicit
! CHECK:         %[[ADDR:.*]] = fir.address_of(@_QFwhole_implicitECc)
! CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ADDR]]
! CHECK:         %[[EXPR:.*]] = hlfir.as_expr %[[DECL]]#0
! CHECK:         %[[TMP:.*]]:3 = hlfir.associate %[[EXPR]]
! CHECK:         fir.call @_QPext(%[[TMP]]#0)
! CHECK:         hlfir.end_associate %[[TMP]]#1, %[[TMP]]#2
subroutine whole_implicit()
  integer, parameter :: c(4) = [1, 2, 3, 4]
  call ext(c)
end subroutine

! Contiguous section of a named constant to an external without an interface.
! CHECK-LABEL: func.func @_QPsection_implicit
! CHECK:         %[[SEC:.*]] = hlfir.designate %{{.*}} (%{{.*}}:%{{.*}}:%{{.*}})
! CHECK:         %[[EXPR:.*]] = hlfir.as_expr %[[SEC]]
! CHECK:         %[[TMP:.*]]:3 = hlfir.associate %[[EXPR]]
! CHECK:         %[[CAST:.*]] = fir.convert %[[TMP]]#0
! CHECK:         fir.call @_QPext(%[[CAST]])
! CHECK:         hlfir.end_associate %[[TMP]]#1, %[[TMP]]#2
subroutine section_implicit()
  integer, parameter :: c(4) = [1, 2, 3, 4]
  call ext(c(2:3))
end subroutine

! Whole named-constant array to an external defined in this file but called
! without an interface: the call is prepared from the actual arguments, as for
! an external defined elsewhere.
! CHECK-LABEL: func.func @_QPwhole_same_file
! CHECK:         %[[EXPR:.*]] = hlfir.as_expr
! CHECK:         %[[TMP:.*]]:3 = hlfir.associate %[[EXPR]]
! CHECK:         fir.call @_QPdefine_it(
subroutine whole_same_file()
  integer, parameter :: c(4) = [1, 2, 3, 4]
  call define_it(c)
end subroutine
subroutine define_it(x)
  integer, intent(out) :: x(4)
  x = 0
end subroutine

! Named-constant array element to an external without an interface: the
! element's address is passed, so that a sequence association sees the rest
! of the named constant.
! CHECK-LABEL: func.func @_QPelement_implicit
! CHECK:         %[[ADDR:.*]] = fir.address_of(@_QFelement_implicitECc)
! CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ADDR]]
! CHECK:         %[[ELT:.*]] = hlfir.designate %[[DECL]]#0 (%{{.*}})
! CHECK-NOT:     hlfir.as_expr
! CHECK-NOT:     hlfir.associate
! CHECK:         %[[CAST:.*]] = fir.convert %[[ELT]]
! CHECK-NOT:     hlfir.associate
! CHECK:         fir.call @_QPext(%[[CAST]])
subroutine element_implicit()
  integer, parameter :: c(4) = [1, 2, 3, 4]
  call ext(c(2))
end subroutine

! Whole named-constant array through an explicit interface: the named
! constant's storage is passed directly.
! CHECK-LABEL: func.func @_QPwhole_explicit
! CHECK:         %[[ADDR:.*]] = fir.address_of(@_QFwhole_explicitECc)
! CHECK:         %[[DECL:.*]]:2 = hlfir.declare %[[ADDR]]
! CHECK-NOT:     hlfir.as_expr
! CHECK-NOT:     hlfir.associate
! CHECK:         fir.call @_QMmPexpl(%[[DECL]]#0)
subroutine whole_explicit()
  use m
  integer, parameter :: c(4) = [1, 2, 3, 4]
  call expl(c)
end subroutine
