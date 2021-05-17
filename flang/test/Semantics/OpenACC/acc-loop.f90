! RUN: %S/../test_errors.sh %s %t %flang -fopenacc

! Check OpenACC clause validity for the following construct and directive:
!   2.9 Loop

program openacc_loop_validity

  implicit none

  type atype
    real(8), dimension(10) :: arr
    real(8) :: s
  end type atype

  integer :: i, j, b, gang_size, vector_size, worker_size
  integer, parameter :: N = 256
  integer, dimension(N) :: c
  logical, dimension(N) :: d, e
  integer :: async1
  integer :: wait1, wait2
  real :: reduction_r
  logical :: reduction_l
  real(8), dimension(N, N) :: aa, bb, cc
  logical :: ifCondition = .TRUE.
  type(atype) :: t
  type(atype), dimension(10) :: ta

  real(8), dimension(N) :: a, f, g, h

  !$acc parallel
  !$acc loop tile(2)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel device_type(*) num_gangs(2)
  !$acc loop
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop seq
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop independent
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop auto
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: At most one VECTOR clause can appear on the LOOP directive
  !$acc loop vector vector(128)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop vector
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop vector(10)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop vector(vector_size)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop vector(length: vector_size)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: At most one WORKER clause can appear on the LOOP directive
  !$acc loop worker worker(10)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop worker
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop worker(10)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop worker(worker_size)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop worker(num: worker_size)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: At most one GANG clause can appear on the LOOP directive
  !$acc loop gang gang(gang_size)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop gang(gang_size)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop gang(num: gang_size)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop gang(gang_size, static:*)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop gang(num: gang_size, static:*)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop gang(num: gang_size, static: gang_size)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop private(b, a(:))
  do i = 1, N
    a(i) = b
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop tile(*)
  do i = 1, N
    a(i) = b
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop tile(2, 2)
  do i = 1, N
    do j = 1, N
      a(i) = b
    end do
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: The parameter of the COLLAPSE clause must be a constant positive integer expression
  !$acc loop collapse(-1)
  do i = 1, N
    do j = 1, N
      a(i) = 3.14 + j
    end do
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Clause PRIVATE is not allowed after clause DEVICE_TYPE on the LOOP directive
  !$acc loop device_type(*) private(i)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Clause GANG is not allowed if clause SEQ appears on the LOOP directive
  !$acc loop gang seq
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Clause WORKER is not allowed if clause SEQ appears on the LOOP directive
  !$acc loop worker seq
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Clause VECTOR is not allowed if clause SEQ appears on the LOOP directive
  !$acc loop vector seq
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !ERROR: Clause IF is not allowed after clause DEVICE_TYPE on the PARALLEL directive
  !$acc parallel device_type(*) if(.TRUE.)
  !$acc loop
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

end program openacc_loop_validity
