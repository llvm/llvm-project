! GOTO exits through two nested ACC data regions. The branch crosses
! two ACC region boundaries, requiring multi-level exit handling.

! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %s -o - 2>&1 | FileCheck %s

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

! CHECK: not yet implemented: GOTO exiting OpenACC region
