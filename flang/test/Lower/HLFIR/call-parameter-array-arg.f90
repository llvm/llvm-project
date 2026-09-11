! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! Test that named constant (PARAMETER) array actual arguments are associated
! with the named constant's storage instead of a temporary copy, when no copy
! is required for argument association (F'2023 15.5.2.12).  A copy is still
! made whenever argument association requires one (non-contiguous actual,
! polymorphic dummy, VALUE dummy, or a true constant expression).

module m
  implicit none
  integer, parameter :: gp(4) = [1, 2, 3, 4]
contains
  subroutine expl(x)
    integer, intent(in) :: x(4)
  end subroutine
  subroutine asmd(x)
    integer, intent(in) :: x(:)
  end subroutine
  subroutine poly(x)
    class(*), intent(in) :: x(:)
  end subroutine
  subroutine byval(x)
    integer, value :: x(4)
  end subroutine
end module

! Whole named-constant array to an explicit-shape dummy: the address of the
! named constant's global is passed directly; no temporary, no copy-in.
! CHECK-LABEL: func.func @_QPwhole_explicit_shape
! CHECK: %[[ADDR:.*]] = fir.address_of(@_QMmECgp) : !fir.ref<!fir.array<4xi32>>
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ADDR]]
! CHECK-NOT: hlfir.as_expr
! CHECK-NOT: hlfir.copy_in
! CHECK: fir.call @_QMmPexpl(%[[DECL]]#0
subroutine whole_explicit_shape()
  use m
  call expl(gp)
end subroutine

! Whole named-constant array to an assumed-shape dummy: a descriptor over the
! named constant's storage; still no temporary.
! CHECK-LABEL: func.func @_QPwhole_assumed_shape
! CHECK: %[[ADDR:.*]] = fir.address_of(@_QMmECgp)
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ADDR]]
! CHECK-NOT: hlfir.as_expr
! CHECK: %[[BOX:.*]] = fir.embox %[[DECL]]#0
! CHECK-NOT: hlfir.as_expr
! CHECK: fir.call @_QMmPasmd
subroutine whole_assumed_shape()
  use m
  call asmd(gp)
end subroutine

! Named-constant array element to an explicit-shape dummy (sequence
! association): the element's address is passed; no scalar temporary.
! CHECK-LABEL: func.func @_QPelement_seq_assoc
! CHECK: %[[EADDR:.*]] = fir.address_of(@_QMmECgp)
! CHECK: %[[EDECL:.*]]:2 = hlfir.declare %[[EADDR]]
! CHECK: %[[ELT:.*]] = hlfir.designate %[[EDECL]]#0 (%{{.*}}) : (!fir.ref<!fir.array<4xi32>>, i64) -> !fir.ref<i32>
! CHECK-NOT: hlfir.as_expr
! CHECK: %[[ECAST:.*]] = fir.convert %[[ELT]] : (!fir.ref<i32>) -> !fir.ref<!fir.array<4xi32>>
! CHECK: fir.call @_QMmPexpl(%[[ECAST]])
subroutine element_seq_assoc()
  use m
  call expl(gp(2))
end subroutine

! Polymorphic assumed-shape dummy: no contiguity requirement, so no copy is
! needed (same as for a variable actual argument); the descriptor is built
! over the named constant's storage.
! CHECK-LABEL: func.func @_QPpoly_dummy
! CHECK: %[[PADDR:.*]] = fir.address_of(@_QMmECgp)
! CHECK: %[[PDECL:.*]]:2 = hlfir.declare %[[PADDR]]
! CHECK-NOT: hlfir.as_expr
! CHECK: %[[PBOX:.*]] = fir.embox %[[PDECL]]#0
! CHECK: %[[PCLS:.*]] = fir.rebox %[[PBOX]] : (!fir.box<!fir.array<4xi32>>) -> !fir.class<!fir.array<?xnone>>
! CHECK: fir.call @_QMmPpoly(%[[PCLS]]
subroutine poly_dummy()
  use m
  call poly(gp)
end subroutine

! A parenthesized named constant is a constant expression, not a designator
! (a parenthesized designator is a primary, R1001): it is still materialized
! into a temporary.
! CHECK-LABEL: func.func @_QPparen_expr
! CHECK: fir.address_of(@_QQro.4xi4.0)
! CHECK: %[[PAREN_TMP:.*]]:3 = hlfir.associate {{.*}} {adapt.valuebyref}
! CHECK: fir.call @_QMmPexpl(%[[PAREN_TMP]]#0
subroutine paren_expr()
  use m
  call expl((gp))
end subroutine

! VALUE dummy: always a copy.
! CHECK-LABEL: func.func @_QPvalue_dummy
! CHECK: hlfir.as_expr
! CHECK: fir.call @_QMmPbyval
subroutine value_dummy()
  use m
  call byval(gp)
end subroutine

! Implicit interface: the whole named-constant array's address is passed
! directly (known contiguous).
! CHECK-LABEL: func.func @_QPimplicit_iface
! CHECK: fir.address_of(@_QMmECgp)
! CHECK-NOT: hlfir.as_expr
! CHECK: fir.call @_QPext_sub
subroutine implicit_iface()
  use m
  external :: ext_sub
  call ext_sub(gp)
end subroutine
