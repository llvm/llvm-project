! RUN: split-file %s %t
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/do_loop_with_stop.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK1
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/do_loop_with_cycle_goto.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK2
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/nested_goto_loop.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK3
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/nested_loop_with_inner_goto.f90 -o - 2>&1 | FileCheck %s --check-prefix=CHECK4

//--- do_loop_with_stop.f90

subroutine do_loop_with_stop()
  integer :: i
  integer, parameter :: n = 10
  real, dimension(n) :: a, b

  !$acc kernels
  do i = 1, n
    a(i) = b(i) + 1.0
    if (i == 5) stop
  end do
  !$acc end kernels

! CHECK1: not yet implemented: unstructured do loop in acc kernels

end subroutine

//--- do_loop_with_cycle_goto.f90

subroutine do_loop_with_cycle_goto()
  integer :: i
  integer, parameter :: n = 10
  real, dimension(n) :: a, b

  ! Do loop with cycle and goto - unstructured control flow is not converted.
  !$acc kernels
  do i = 1, n
    if (i == 3) cycle
    a(i) = b(i) + 1.0
    if (i == 7) goto 200
    a(i) = a(i) * 2.0
  end do
200 continue
  !$acc end kernels

! CHECK2: not yet implemented: unstructured do loop in acc kernels

end subroutine

//--- nested_goto_loop.f90

subroutine nested_goto_loop()
  integer :: i, j
  integer, parameter :: n = 10, m = 5
  real, dimension(n,m) :: a, b

  ! Nested loop with goto from inner to outer - should NOT convert to acc.loop
  !$acc kernels
  do i = 1, n
    do j = 1, m
      a(i,j) = b(i,j) + 1.0
      if (i * j > 20) goto 300  ! Exit both loops
    end do
  end do
300 continue
  !$acc end kernels

! CHECK3: not yet implemented: unstructured do loop in acc kernels

end subroutine

//--- nested_loop_with_inner_goto.f90

subroutine nested_loop_with_inner_goto()
  integer :: ii = 0, jj = 0
  integer, parameter :: nn = 3
  real, dimension(nn, nn) :: aa

  aa = -1

  ! Nested loop with goto from inner loop - unstructured control flow is not converted.
  !$acc kernels
  do ii = 1, nn
    do jj = 1, nn
      if (jj > 1) goto 300
      aa(jj, ii) = 1337
    end do
    300 continue
  end do
  !$acc end kernels

! CHECK4: not yet implemented: unstructured do loop in acc kernels

end subroutine
