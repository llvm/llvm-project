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

! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

module m
  type t
    integer :: n
  end type t
  type t2
    ! Parenthesized component default initializer.
    integer :: n = (5)
  end type t2

  integer :: i = (42)
  real :: r = (3.5)
  logical :: l = (.true.)
  ! Double parentheses so the initializer exercises a Parentheses node rather
  ! than plain complex-literal syntax.
  complex :: z = ((1.0, 2.0))
  ! Parenthesized structure constructor.
  type(t) :: x = (t(7))
  ! Default-initialized object exercising the parenthesized component default.
  type(t2) :: w

  ! Belt-and-braces: parentheses stripped before lowering.
  character(2) :: c = ('ab')
  integer, parameter :: iparm(2) = [1, 2]
  integer :: a(2) = (iparm)
  type(t) :: y = t((5))
end module m

! Globals are emitted in the order: a, c, i, iparm, l, r, w, x, y, z.

! Parenthesized array named-constant: parentheses stripped, dense global.
! CHECK: fir.global @_QMmEa(dense<[1, 2]> : tensor<2xi32>) {{.*}} : !fir.array<2xi32>

! Character: parentheses stripped during initializer conversion, plain string.
! CHECK-LABEL: fir.global @_QMmEc : !fir.char<1,2> {
! CHECK:         %[[S:.*]] = fir.string_lit "ab"(2) : !fir.char<1,2>
! CHECK-NOT:     fir.no_reassoc
! CHECK:         fir.has_value %[[S]] : !fir.char<1,2>

! CHECK-LABEL: fir.global @_QMmEi : i32 {
! CHECK:         %[[C:.*]] = arith.constant 42 : i32
! CHECK:         %[[NR:.*]] = fir.no_reassoc %[[C]] : i32
! CHECK:         fir.has_value %[[NR]] : i32

! CHECK-LABEL: fir.global @_QMmEl : !fir.logical<4> {
! CHECK:         %[[C:.*]] = arith.constant true
! CHECK:         %[[NR:.*]] = fir.no_reassoc %[[C]] : i1
! CHECK:         %[[CV:.*]] = fir.convert %[[NR]] : (i1) -> !fir.logical<4>
! CHECK:         fir.has_value %[[CV]] : !fir.logical<4>

! CHECK-LABEL: fir.global @_QMmEr : f32 {
! CHECK:         %[[C:.*]] = arith.constant 3.500000e+00 : f32
! CHECK:         %[[NR:.*]] = fir.no_reassoc %[[C]] : f32
! CHECK:         fir.has_value %[[NR]] : f32

! Parenthesized component default: the parentheses wrap the component value in a
! fir.no_reassoc before it is inserted.
! CHECK-LABEL: fir.global @_QMmEw : !fir.type<_QMmTt2{n:i32}> {
! CHECK:         %[[C:.*]] = arith.constant 5 : i32
! CHECK:         %[[NR:.*]] = fir.no_reassoc %[[C]] : i32
! CHECK:         %[[IV:.*]] = fir.insert_value %{{.*}}, %[[NR]], ["n", !fir.type<_QMmTt2{n:i32}>]
! CHECK:         fir.has_value %[[IV]] : !fir.type<_QMmTt2{n:i32}>

! Parenthesized structure constructor: the parentheses survive folding and wrap
! the insert_value chain in a fir.no_reassoc.
! CHECK-LABEL: fir.global @_QMmEx : !fir.type<_QMmTt{n:i32}> {
! CHECK:         %[[IV:.*]] = fir.insert_value %{{.*}}, %{{.*}}, ["n", !fir.type<_QMmTt{n:i32}>]
! CHECK:         %[[NR:.*]] = fir.no_reassoc %[[IV]] : !fir.type<_QMmTt{n:i32}>
! CHECK:         fir.has_value %[[NR]] : !fir.type<_QMmTt{n:i32}>

! Parenthesized structure-constructor component: parentheses stripped, plain
! insert_value chain with no fir.no_reassoc.
! CHECK-LABEL: fir.global @_QMmEy : !fir.type<_QMmTt{n:i32}> {
! CHECK:         %[[IV:.*]] = fir.insert_value %{{.*}}, %{{.*}}, ["n", !fir.type<_QMmTt{n:i32}>]
! CHECK-NOT:     fir.no_reassoc
! CHECK:         fir.has_value %[[IV]] : !fir.type<_QMmTt{n:i32}>

! CHECK-LABEL: fir.global @_QMmEz : complex<f32> {
! CHECK:         fir.insert_value
! CHECK:         %[[NR:.*]] = fir.no_reassoc %{{.*}} : complex<f32>
! CHECK:         fir.has_value %[[NR]] : complex<f32>
