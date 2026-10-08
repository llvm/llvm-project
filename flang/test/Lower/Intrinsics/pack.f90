! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPpack_test(
! CHECK-SAME: %[[arg0:[^:]+]]: !fir.box<!fir.array<?xi32>>
! CHECK-SAME: %[[arg1:.*]]: !fir.box<!fir.array<?x!fir.logical<4>>>
! CHECK-SAME: %[[arg2:[^:]+]]: !fir.box<!fir.array<?xi32>>
! CHECK-SAME: %[[arg3:[^:]+]]: !fir.box<!fir.array<?xi32>>
subroutine pack_test(a,m,v,r)
    integer :: a(:)
    logical :: m(:)
    integer :: v(:)
    integer :: r(:)
! CHECK-DAG:  %[[A:.*]]:2 = hlfir.declare %[[arg0]]
! CHECK-DAG:  %[[M:.*]]:2 = hlfir.declare %[[arg1]]
! CHECK-DAG:  %[[V:.*]]:2 = hlfir.declare %[[arg2]]
  r = pack(a,m,v)
! CHECK:  hlfir.pack
! CHECK-NOT: fir.call @_FortranAPack
  end subroutine

  ! CHECK-LABEL: func.func @_QPtest_pack_optional(
  ! CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
  subroutine test_pack_optional(vector, array, mask)
    integer, pointer :: vector(:)
    integer :: array(:, :)
    logical :: mask(:, :)
    print *, pack(array, mask, vector)
  ! CHECK:  hlfir.pack
  ! CHECK-NOT: fir.call @_FortranAPack
  end subroutine
