! Test lowering of parenthesized initial values in fir.global initializer
! regions.
!
! Initial values are lowered through ConvertConstant, which folds a
! parenthesized scalar or derived constant to a plain constant: no
! fir.no_reassoc operation is emitted inside the global init region. (An earlier
! baseline of this test pinned the previous behavior, where a parenthesized
! constant survived folding as a Parentheses node and lowered to a
! fir.no_reassoc.) The forms whose parentheses are stripped before lowering
! (character, array named-constant, and parenthesized structure-constructor
! components) are unaffected and pinned here belt-and-braces.
!
! Each case is a SAVE'd local in its own subroutine so that the fir.global for
! it is emitted in source order, keeping every CHECK block next to the Fortran
! it checks. The --implicit-check-not on the RUN line asserts that no
! fir.no_reassoc is emitted for any of the parenthesized initializers.

! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --implicit-check-not=fir.no_reassoc

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
! CHECK:         fir.has_value %[[C]] : i32

subroutine nested_parens()
  ! Stacked parentheses: exercises the recursive descent through nested
  ! Parentheses<T> nodes.
  integer, save :: n2 = ((42))
end subroutine
! CHECK-LABEL: fir.global internal @_QFnested_parensEn2 : i32 {
! CHECK:         %[[C:.*]] = arith.constant 42 : i32
! CHECK:         fir.has_value %[[C]] : i32

subroutine scalar_real()
  real, save :: r = (3.5)
end subroutine
! CHECK-LABEL: fir.global internal @_QFscalar_realEr : f32 {
! CHECK:         %[[C:.*]] = arith.constant 3.500000e+00 : f32
! CHECK:         fir.has_value %[[C]] : f32

subroutine scalar_logical()
  logical, save :: l = (.true.)
end subroutine
! CHECK-LABEL: fir.global internal @_QFscalar_logicalEl : !fir.logical<4> {
! CHECK:         %[[C:.*]] = arith.constant true
! CHECK:         %[[CV:.*]] = fir.convert %[[C]] : (i1) -> !fir.logical<4>
! CHECK:         fir.has_value %[[CV]] : !fir.logical<4>

subroutine scalar_complex()
  ! Double parentheses so the initializer exercises a Parentheses node rather
  ! than plain complex-literal syntax.
  complex, save :: z = ((1.0, 2.0))
end subroutine
! CHECK-LABEL: fir.global internal @_QFscalar_complexEz : complex<f32> {
! CHECK:         fir.insert_value
! CHECK:         %[[IV:.*]] = fir.insert_value
! CHECK:         fir.has_value %[[IV]] : complex<f32>

subroutine paren_ctor()
  use types
  ! Parenthesized structure constructor: folded to a plain insert_value chain.
  type(t), save :: x = (t(7))
end subroutine
! CHECK-LABEL: fir.global internal @_QFparen_ctorEx : !fir.type<_QMtypesTt{n:i32}> {
! CHECK:         %[[IV:.*]] = fir.insert_value %{{.*}}, %{{.*}}, ["n", !fir.type<_QMtypesTt{n:i32}>]
! CHECK:         fir.has_value %[[IV]] : !fir.type<_QMtypesTt{n:i32}>

subroutine paren_derived_named_const()
  use types
  ! Parenthesized derived named constant: the parentheses wrap a
  ! Constant<SomeDerived> rather than a structure constructor.
  type(t), parameter :: tp = t(3)
  type(t), save :: pt2 = (tp)
end subroutine
! (The named constant tp also gets its own constant global; its position
! relative to pt2's global is unpinned.)
! CHECK-LABEL: fir.global internal @_QFparen_derived_named_constEpt2 : !fir.type<_QMtypesTt{n:i32}> {
! CHECK:         %[[C:.*]] = arith.constant 3 : i32
! CHECK:         %[[IV:.*]] = fir.insert_value %{{.*}}, %[[C]], ["n", !fir.type<_QMtypesTt{n:i32}>]
! CHECK:         fir.has_value %[[IV]] : !fir.type<_QMtypesTt{n:i32}>

subroutine comp_default()
  use types
  ! Default-initialized object exercising the parenthesized component default:
  ! folded to a plain component value.
  type(t2), save :: w
end subroutine
! CHECK-LABEL: fir.global internal @_QFcomp_defaultEw : !fir.type<_QMtypesTt2{n:i32}> {
! CHECK:         %[[C:.*]] = arith.constant 5 : i32
! CHECK:         %[[IV:.*]] = fir.insert_value %{{.*}}, %[[C]], ["n", !fir.type<_QMtypesTt2{n:i32}>]
! CHECK:         fir.has_value %[[IV]] : !fir.type<_QMtypesTt2{n:i32}>

subroutine char_paren()
  ! Belt-and-braces: parentheses stripped before lowering.
  character(2), save :: c = ('ab')
end subroutine
! CHECK-LABEL: fir.global internal @_QFchar_parenEc : !fir.char<1,2> {
! CHECK:         %[[S:.*]] = fir.string_lit "ab"(2) : !fir.char<1,2>
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
  ! insert_value chain.
  type(t), save :: y = t((5))
end subroutine
! CHECK-LABEL: fir.global internal @_QFparen_ctor_compEy : !fir.type<_QMtypesTt{n:i32}> {
! CHECK:         %[[IV:.*]] = fir.insert_value %{{.*}}, %{{.*}}, ["n", !fir.type<_QMtypesTt{n:i32}>]
! CHECK:         fir.has_value %[[IV]] : !fir.type<_QMtypesTt{n:i32}>
