! RUN: %python %S/../test_errors.py %s %flang -fopenacc -Werror

! OpenACC 3.4 2.15.1: a routine may parent a loop at its level or below.
! A higher level is not permitted and may be ignored. 2.9 uses the same order for nested loops.

subroutine worker_ignores_gang(a)
  real :: a(10)
  integer :: i
  !$acc routine worker
  !WARNING: GANG clause on the LOOP directive is not permitted and may be ignored in ACC ROUTINE WORKER procedure [-Wopenacc-usage]
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
  !WARNING: GANG clause on the LOOP directive is not permitted and may be ignored in ACC ROUTINE VECTOR procedure [-Wopenacc-usage]
  !$acc loop gang
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
  !WARNING: WORKER clause on the LOOP directive is not permitted and may be ignored in ACC ROUTINE VECTOR procedure [-Wopenacc-usage]
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
  !WARNING: GANG clause on the LOOP directive is not permitted and may be ignored in ACC ROUTINE SEQ procedure [-Wopenacc-usage]
  !$acc loop gang
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
  !WARNING: WORKER clause on the LOOP directive is not permitted and may be ignored in ACC ROUTINE SEQ procedure [-Wopenacc-usage]
  !$acc loop worker
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
  !WARNING: VECTOR clause on the LOOP directive is not permitted and may be ignored in ACC ROUTINE SEQ procedure [-Wopenacc-usage]
  !$acc loop vector
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
end subroutine

subroutine gang1_ignores_gang2(a)
  real :: a(10)
  integer :: i
  !$acc routine gang(dim:1)
  !WARNING: GANG(2) clause on the LOOP directive is not permitted and may be ignored in ACC ROUTINE GANG(1) procedure [-Wopenacc-usage]
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
  !WARNING: GANG(3) clause on the LOOP directive is not permitted and may be ignored in ACC ROUTINE GANG(2) procedure [-Wopenacc-usage]
  !$acc loop gang(dim:3)
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
end subroutine

subroutine device_type_worker_ignores_gang(a)
  real :: a(10)
  integer :: i
  !$acc routine device_type(nvidia) worker
  !WARNING: GANG clause on the LOOP directive is not permitted and may be ignored in ACC ROUTINE WORKER procedure for DEVICE_TYPE(NVIDIA) [-Wopenacc-usage]
  !$acc loop gang worker vector
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
end subroutine

subroutine device_types_each_ignore_gang(a)
  real :: a(10)
  integer :: i
  !$acc routine device_type(nvidia) worker device_type(radeon) worker
  !WARNING: GANG clause on the LOOP directive is not permitted and may be ignored in ACC ROUTINE WORKER procedure for DEVICE_TYPE(NVIDIA) [-Wopenacc-usage]
  !WARNING: GANG clause on the LOOP directive is not permitted and may be ignored in ACC ROUTINE WORKER procedure for DEVICE_TYPE(RADEON) [-Wopenacc-usage]
  !$acc loop device_type(nvidia) gang device_type(radeon) gang
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
end subroutine

subroutine star_vector_ignores_worker(a)
  real :: a(10)
  integer :: i
  !$acc routine device_type(*) vector
  !WARNING: WORKER clause on the LOOP directive is not permitted and may be ignored in ACC ROUTINE VECTOR procedure for DEVICE_TYPE(*) [-Wopenacc-usage]
  !$acc loop worker
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
end subroutine

! The loop device_type(*) gang is compared with the routine's
! device_type(*) gang. The NVIDIA worker level is not applied.
subroutine star_gang_does_not_use_nvidia_worker(a)
  real :: a(10)
  integer :: i
  !$acc routine device_type(nvidia) worker device_type(*) gang
  !$acc loop device_type(*) gang
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
end subroutine

subroutine device_gang_dim_uses_current_clause(a)
  real :: a(10)
  integer :: i
  !$acc routine gang(dim:1)
  !WARNING: GANG(2) clause on the LOOP directive is not permitted and may be ignored in ACC ROUTINE GANG(1) procedure [-Wopenacc-usage]
  !$acc loop gang(dim:1) device_type(nvidia) gang(dim:2)
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
end subroutine

subroutine star_gang_exceeds_star_worker(a)
  real :: a(10)
  integer :: i
  !$acc routine device_type(*) worker
  !WARNING: GANG clause on the LOOP directive is not permitted and may be ignored in ACC ROUTINE WORKER procedure for DEVICE_TYPE(*) [-Wopenacc-usage]
  !$acc loop device_type(*) gang
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
end subroutine

subroutine nvidia_gang_exceeds_seq(a)
  real :: a(10)
  integer :: i
  !$acc routine device_type(nvidia) seq
  ! The default gang is covered by the NVIDIA gang, so only that gang warns.
  !$acc loop gang &
  !WARNING: GANG clause on the LOOP directive is not permitted and may be ignored in ACC ROUTINE SEQ procedure for DEVICE_TYPE(NVIDIA) [-Wopenacc-usage]
  !$acc& device_type(nvidia) gang
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
end subroutine

! Two routine directives keep both levels. Each loop gang matches its level.
subroutine default_gang_dim_overridden_for_nvidia(a)
  real :: a(10)
  integer :: i
  !$acc routine gang(dim:2)
  !$acc routine device_type(nvidia) gang(dim:1)
  !$acc loop gang(dim:2) device_type(nvidia) gang(dim:1)
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
end subroutine

! A later device-specific gang is the level used for the NVIDIA loop.
! The earlier vector level is not also applied.
subroutine device_gang_does_not_keep_vector(a)
  real :: a(10)
  integer :: i
  !$acc routine vector
  !$acc routine device_type(nvidia) gang
  !$acc loop device_type(nvidia) worker
  do i = 1, 10
    a(i) = a(i) + 1.0
  end do
end subroutine
