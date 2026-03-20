! RUN: split-file %s %t
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/acc_loop_multi_level.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK1
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/acc_data_multi_level.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK2

//--- acc_loop_multi_level.f90

subroutine acc_loop_multi_level(a, n)
  integer :: n, i, j
  real :: a(*)

  !$acc parallel
  !$acc loop seq
  do i = 1, n
    do j = 1, n
      if (a(j) > 0.0) goto 999
    end do
  end do
  !$acc end parallel
999 continue
end subroutine

! CHECK1: not yet implemented: GOTO exiting OpenACC region

//--- acc_data_multi_level.f90

subroutine acc_data_multi_level(a, n)
  integer :: n, i, j
  real :: a(*)

  !$acc parallel
  !$acc data
  do i = 1, n
    do j = 1, n
      if (a(j) > 0.0) goto 999
    end do
  end do
  !$acc end data
  !$acc end parallel
999 continue
end subroutine

! CHECK2: not yet implemented: GOTO exiting OpenACC region
