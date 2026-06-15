! Each sub-file exercises a different unstructured-CFG pattern inside a
! combined `acc parallel loop` construct (default parallelism is
! `independent`).

! RUN: split-file %s %t
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/stop_collapse1.f90 -o - 2>&1 | FileCheck %s --check-prefix=STOP1
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/cycle_collapse2.f90 -o - 2>&1 | FileCheck %s --check-prefix=CYCLE2
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir %t/stop_collapse3.f90 -o - 2>&1 | FileCheck %s --check-prefix=STOP3

!--- stop_collapse1.f90

! `acc parallel loop` with STOP in the body. Loop defaults to `independent`.
subroutine test_unstructured2(a, b, c)
  integer :: i, j, k
  real :: a(:,:,:), b(:,:,:), c(:,:,:)

  !$acc parallel loop
  do i = 1, 10
    do j = 1, 10
      do k = 1, 10
        if (a(1,2,3) > 10) stop 'just to be unstructured'
      end do
    end do
  end do

end subroutine

! STOP1: not yet implemented: unstructured do loop in combined acc construct

!--- cycle_collapse2.f90

! `acc parallel loop collapse(2)` with an early-exit (CYCLE).
subroutine test_unstructured_collapse_cycle(a)
  integer :: i, j, jdiag
  real(8) :: a(:,:)
  jdiag = 4
  !$acc parallel loop collapse(2) copy(a)
  do j = 1, 8
    do i = 1, 8
      if (i == jdiag) then
        a(i, j) = 0.0d0
        cycle
      end if
      a(i, j) = real(i + j, 8)
    end do
  end do
  !$acc end parallel loop
end subroutine

! CYCLE2: not yet implemented: unstructured do loop in combined acc construct

!--- stop_collapse3.f90

! `acc parallel loop collapse(3)` with STOP - the collapse=3 form of the
! STOP scenario above.
subroutine test_unstructured_collapse_stop(a)
  integer :: i, j, k
  real :: a(:,:,:)
  !$acc parallel loop collapse(3)
  do i = 1, 10
    do j = 1, 10
      do k = 1, 10
        if (a(1,2,3) > 10) stop 'just to be unstructured'
      end do
    end do
  end do
end subroutine

! STOP3: not yet implemented: unstructured do loop in combined acc construct
