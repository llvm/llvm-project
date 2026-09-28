! RUN: %python %S/../test_errors.py %s %flang -fopenacc -Werror

! OpenACC 3.4 2.15.1: a routine may parent a loop at its level or below.
! A higher level is ignored. 2.9 uses the same order for nested loops.

subroutine worker_ignores_gang(a)
  real :: a(10)
  integer :: i
  !$acc routine worker
  !WARNING: GANG clause ignored in ACC ROUTINE WORKER procedure [-Wopenacc-usage]
  !$acc loop gang worker vector
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
end subroutine

subroutine worker_keeps_worker_vector(a)
  real :: a(10)
  integer :: i
  !$acc routine worker
  !$acc loop worker vector
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
end subroutine

subroutine vector_ignores_gang_and_worker(a)
  real :: a(10)
  integer :: i
  !$acc routine vector
  !WARNING: GANG clause ignored in ACC ROUTINE VECTOR procedure [-Wopenacc-usage]
  !$acc loop gang
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
  !WARNING: WORKER clause ignored in ACC ROUTINE VECTOR procedure [-Wopenacc-usage]
  !$acc loop worker
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
  !$acc loop vector
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
end subroutine

subroutine seq_ignores_all(a)
  real :: a(10)
  integer :: i
  !$acc routine seq
  !WARNING: GANG clause ignored in ACC ROUTINE SEQ procedure [-Wopenacc-usage]
  !$acc loop gang
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
  !WARNING: WORKER clause ignored in ACC ROUTINE SEQ procedure [-Wopenacc-usage]
  !$acc loop worker
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
  !WARNING: VECTOR clause ignored in ACC ROUTINE SEQ procedure [-Wopenacc-usage]
  !$acc loop vector
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
end subroutine

subroutine gang1_ignores_gang2(a)
  real :: a(10)
  integer :: i
  !$acc routine gang(dim:1)
  !WARNING: GANG(2) clause ignored in ACC ROUTINE GANG(1) procedure [-Wopenacc-usage]
  !$acc loop gang(dim:2)
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
  !$acc loop gang
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
  !$acc loop worker vector
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
end subroutine

subroutine gang2_keeps_gang1(a)
  real :: a(10)
  integer :: i
  !$acc routine gang(dim:2)
  !$acc loop gang(dim:1)
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
  !WARNING: GANG(3) clause ignored in ACC ROUTINE GANG(2) procedure [-Wopenacc-usage]
  !$acc loop gang(dim:3)
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
end subroutine

subroutine device_type_worker_ignores_gang(a)
  real :: a(10)
  integer :: i
  !$acc routine device_type(nvidia) worker
  !WARNING: GANG clause ignored in ACC ROUTINE WORKER procedure for DEVICE_TYPE(NVIDIA) [-Wopenacc-usage]
  !$acc loop gang worker vector
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
end subroutine

subroutine star_vector_ignores_worker(a)
  real :: a(10)
  integer :: i
  !$acc routine device_type(*) vector
  !WARNING: WORKER clause ignored in ACC ROUTINE VECTOR procedure for DEVICE_TYPE(*) [-Wopenacc-usage]
  !$acc loop worker
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
end subroutine
