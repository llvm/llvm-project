! GOTO exits through two nested ACC regions. The branch crosses
! two ACC region boundaries, requiring multi-level exit handling.

! RUN: split-file %s %t
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/nested_data.f90 -o - 2>&1 | FileCheck %s --check-prefix=DATA
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/nested_loop.f90 -o - 2>&1 | FileCheck %s --check-prefix=LOOP

!--- nested_data.f90

subroutine nested_data_exit(a, n)
  integer :: n, i, j
  real :: a(*)

  !$acc data copy(a(1:n))
  !$acc data copyout(a(1:n))
  do i = 1, n
    do j = 1, n
      if (a(j) > 0.0) goto 888
    end do
  end do
  !$acc end data
  !$acc end data
888 continue
end subroutine

! DATA: not yet implemented: GOTO exiting OpenACC region

!--- nested_loop.f90

subroutine nested_loop_exit(A, B, N)
  implicit real*8 (a-h, o-z)
  !$acc routine seq
  dimension A(*), B(*)
  !$acc loop seq
  do 100 i = 1, N
  !$acc loop seq
    do 10 j = 1, N
      if (A(j) .gt. B(j)) goto 200
10  continue
100 continue
200 continue
end subroutine

! LOOP: not yet implemented: GOTO exiting OpenACC region
