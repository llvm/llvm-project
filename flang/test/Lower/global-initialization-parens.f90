! Test lowering of parenthesized initial values in fir.global initializer
! regions.
!
! This test baselines the *current* lowering behavior of parenthesized
! initializers, which an upcoming change to the initializer-lowering path will
! modify: today a parenthesized scalar or derived constant survives folding as
! a Parentheses node and is lowered to a fir.no_reassoc operation inside the
! global init region. The forms whose parentheses are stripped before lowering
! (character, array named-constant, and parenthesized structure-constructor
! components) are pinned here belt-and-braces so the follow-up change is shown
! to leave them untouched.
!
! Each case is a SAVE'd local in its own subroutine so that the fir.global for
! it is emitted in source order, keeping every CHECK block next to the Fortran
! it checks.

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

module types
  type t
    integer :: n
  end type t
  type t2
    ! Parenthesized component default initializer.
    integer :: n = (5)
  end type t2
end module types

subroutine scalar_int()
  integer, save :: i = (42)
end subroutine
! CHECK-LABEL: fir.global internal @_QFscalar_intEi : i32 {
! CHECK:         %[[C:.*]] = arith.constant 42 : i32
! CHECK:         %[[NR:.*]] = fir.no_reassoc %[[C]] : i32
! CHECK:         fir.has_value %[[NR]] : i32

subroutine scalar_real()
  real, save :: r = (3.5)
end subroutine
! CHECK-LABEL: fir.global internal @_QFscalar_realEr : f32 {
! CHECK:         %[[C:.*]] = arith.constant 3.500000e+00 : f32
! CHECK:         %[[NR:.*]] = fir.no_reassoc %[[C]] : f32
! CHECK:         fir.has_value %[[NR]] : f32

subroutine scalar_logical()
  logical, save :: l = (.true.)
end subroutine
! CHECK-LABEL: fir.global internal @_QFscalar_logicalEl : !fir.logical<4> {
! CHECK:         %[[C:.*]] = arith.constant true
! CHECK:         %[[NR:.*]] = fir.no_reassoc %[[C]] : i1
! CHECK:         %[[CV:.*]] = fir.convert %[[NR]] : (i1) -> !fir.logical<4>
! CHECK:         fir.has_value %[[CV]] : !fir.logical<4>

subroutine scalar_complex()
  ! Double parentheses so the initializer exercises a Parentheses node rather
  ! than plain complex-literal syntax.
  complex, save :: z = ((1.0, 2.0))
end subroutine
! CHECK-LABEL: fir.global internal @_QFscalar_complexEz : complex<f32> {
! CHECK:         fir.insert_value
! CHECK:         %[[NR:.*]] = fir.no_reassoc %{{.*}} : complex<f32>
! CHECK:         fir.has_value %[[NR]] : complex<f32>

subroutine paren_ctor()
  use types
  ! Parenthesized structure constructor: the parentheses survive folding and
  ! wrap the insert_value chain in a fir.no_reassoc.
  type(t), save :: x = (t(7))
end subroutine
! CHECK-LABEL: fir.global internal @_QFparen_ctorEx : !fir.type<_QMtypesTt{n:i32}> {
! CHECK:         %[[IV:.*]] = fir.insert_value %{{.*}}, %{{.*}}, ["n", !fir.type<_QMtypesTt{n:i32}>]
! CHECK:         %[[NR:.*]] = fir.no_reassoc %[[IV]] : !fir.type<_QMtypesTt{n:i32}>
! CHECK:         fir.has_value %[[NR]] : !fir.type<_QMtypesTt{n:i32}>

subroutine comp_default()
  use types
  ! Default-initialized object exercising the parenthesized component default:
  ! the parentheses wrap the component value in a fir.no_reassoc before it is
  ! inserted.
  type(t2), save :: w
end subroutine
! CHECK-LABEL: fir.global internal @_QFcomp_defaultEw : !fir.type<_QMtypesTt2{n:i32}> {
! CHECK:         %[[C:.*]] = arith.constant 5 : i32
! CHECK:         %[[NR:.*]] = fir.no_reassoc %[[C]] : i32
! CHECK:         %[[IV:.*]] = fir.insert_value %{{.*}}, %[[NR]], ["n", !fir.type<_QMtypesTt2{n:i32}>]
! CHECK:         fir.has_value %[[IV]] : !fir.type<_QMtypesTt2{n:i32}>

subroutine char_paren()
  ! Belt-and-braces: parentheses stripped before lowering.
  character(2), save :: c = ('ab')
end subroutine
! CHECK-LABEL: fir.global internal @_QFchar_parenEc : !fir.char<1,2> {
! CHECK:         %[[S:.*]] = fir.string_lit "ab"(2) : !fir.char<1,2>
! CHECK-NOT:     fir.no_reassoc
! CHECK:         fir.has_value %[[S]] : !fir.char<1,2>

subroutine array_named_const()
  ! Parenthesized array named-constant: parentheses stripped, dense global.
  integer, parameter :: iparm(2) = [1, 2]
  integer, save :: a(2) = (iparm)
end subroutine
! CHECK: fir.global internal @_QFarray_named_constEa(dense<[1, 2]> : tensor<2xi32>) {{.*}} : !fir.array<2xi32>

subroutine paren_ctor_comp()
  use types
  ! Parenthesized structure-constructor component: parentheses stripped, plain
  ! insert_value chain with no fir.no_reassoc.
  type(t), save :: y = t((5))
end subroutine
! CHECK-LABEL: fir.global internal @_QFparen_ctor_compEy : !fir.type<_QMtypesTt{n:i32}> {
! CHECK:         %[[IV:.*]] = fir.insert_value %{{.*}}, %{{.*}}, ["n", !fir.type<_QMtypesTt{n:i32}>]
! CHECK-NOT:     fir.no_reassoc
! CHECK:         fir.has_value %[[IV]] : !fir.type<_QMtypesTt{n:i32}>
