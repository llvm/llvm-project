! Test character_pointer => NULL()
! The main point is to check that non deferred length parameter is preserved
! inside the descriptor, and that the length is otherwise set to zero.
! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

subroutine test_deferred(p)
  character(:), pointer :: p
  p => null()
end subroutine
subroutine test_cst(p)
  character(10), pointer :: p
  p => null()
end subroutine
subroutine test_explicit(p, n)
  integer(8) :: n
  character(n), pointer :: p
  p => null()
end subroutine
subroutine test_assumed(p)
  character(*), pointer :: p
  p => null()
end subroutine
subroutine test_deferred_comp(p)
  type t
    character(:), pointer :: p
  end type
  type(t) :: x
  x%p => null()
end subroutine
subroutine test_explicit_comp(p)
  type t
    character(10), pointer :: p
  end type
  type(t) :: x
  x%p => null()
end subroutine

! CHECK-LABEL:   func.func @_QPtest_deferred(
! CHECK:           %[[ZERO_BITS_0:.*]] = fir.zero_bits !fir.ptr<!fir.char<1,?>>
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ZERO_BITS_0]] typeparams %[[CONSTANT_0]] : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>

! CHECK-LABEL:   func.func @_QPtest_cst(
! CHECK:           %[[ZERO_BITS_0:.*]] = fir.zero_bits !fir.ptr<!fir.char<1,10>>
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ZERO_BITS_0]] : (!fir.ptr<!fir.char<1,10>>) -> !fir.box<!fir.ptr<!fir.char<1,10>>>

! CHECK-LABEL:   func.func @_QPtest_explicit(
! CHECK:           %[[LOAD_0:.*]] = fir.load %{{.*}} : !fir.ref<i64>
! CHECK:           %[[SELECT_0:.*]] = arith.select %{{.*}}, %[[LOAD_0]], %c0{{.*}} : i64
! CHECK:           %[[ZERO_BITS_0:.*]] = fir.zero_bits !fir.ptr<!fir.char<1,?>>
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ZERO_BITS_0]] typeparams %[[SELECT_0]] : (!fir.ptr<!fir.char<1,?>>, i64) -> !fir.box<!fir.ptr<!fir.char<1,?>>>

! CHECK-LABEL:   func.func @_QPtest_assumed(
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[ARG0:.*]] : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>
! CHECK:           %[[BOX_ELESIZE_0:.*]] = fir.box_elesize %[[LOAD_0]] : (!fir.box<!fir.ptr<!fir.char<1,?>>>) -> index
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK:           %[[ZERO_BITS_0:.*]] = fir.zero_bits !fir.ptr<!fir.char<1,?>>
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ZERO_BITS_0]] typeparams %[[BOX_ELESIZE_0]] : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>

! CHECK-LABEL:   func.func @_QPtest_deferred_comp(
! CHECK:           %[[ZERO_BITS_0:.*]] = fir.zero_bits !fir.ptr<!fir.char<1,?>>
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ZERO_BITS_0]] typeparams %[[CONSTANT_0]] : (!fir.ptr<!fir.char<1,?>>, index) -> !fir.box<!fir.ptr<!fir.char<1,?>>>
! CHECK:           fir.store %[[EMBOX_0]] to %{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,?>>>>

! CHECK-LABEL:   func.func @_QPtest_explicit_comp(
! CHECK:           %[[ZERO_BITS_0:.*]] = fir.zero_bits !fir.ptr<!fir.char<1,10>>
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ZERO_BITS_0]] : (!fir.ptr<!fir.char<1,10>>) -> !fir.box<!fir.ptr<!fir.char<1,10>>>
! CHECK:           fir.store %[[EMBOX_0]] to %{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.char<1,10>>>>
