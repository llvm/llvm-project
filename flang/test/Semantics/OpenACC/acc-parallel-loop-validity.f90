! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Check OpenACC clause validity for the following construct and directive:
!   2.11 Parallel Loop

program openacc_parallel_loop_validity

  implicit none

  integer :: i, j, b
  integer, parameter :: N = 256
  integer, dimension(N) :: c
  logical, dimension(N) :: d, e
  real :: reduction_r
  logical :: reduction_l
  logical :: ifCondition = .TRUE.
  real(8), dimension(N) :: a, f, g, h
  real(8), dimension(N, N) :: aa, bb, cc

  !$acc parallel loop
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc parallel loop
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel loop

  !$acc parallel loop
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !$acc parallel loop tile(2)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc parallel loop self
  do i = 1, N
    a(i) = 3.14
  end do

  !ERROR: SELF clause on the PARALLEL LOOP directive only accepts optional scalar logical expression
  !$acc parallel loop self(bb, cc(:,:))
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc parallel loop self(.true.)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc parallel loop self(ifCondition)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc parallel loop tile(2, 2)
  do i = 1, N
    do j = 1, N
      aa(i, j) = 3.14
    end do
  end do

  !ERROR: Clause IF is not allowed after clause DEVICE_TYPE on the PARALLEL LOOP directive
  !$acc parallel loop device_type(*) if(.TRUE.)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel loop

  !$acc kernels loop
  do i = 1, N
    a(i) = 3.14
  end do
  !ERROR: Unmatched END PARALLEL LOOP directive
  !$acc end parallel loop

  !$acc parallel loop reduction(+: reduction_r)
  do i = 1, N
    reduction_r = a(i) + i
  end do

  !$acc parallel loop reduction(*: reduction_r)
  do i = 1, N
    reduction_r = reduction_r * (a(i) + i)
  end do

  !$acc parallel loop reduction(min: reduction_r)
  do i = 1, N
    reduction_r = min(reduction_r, a(i) * i)
  end do

  !$acc parallel loop reduction(max: reduction_r)
  do i = 1, N
    reduction_r = max(reduction_r, a(i) * i)
  end do

  !$acc parallel loop reduction(iand: b)
  do i = 1, N
    b = iand(b, c(i))
  end do

  !$acc parallel loop reduction(ior: b)
  do i = 1, N
    b = ior(b, c(i))
  end do

  !$acc parallel loop reduction(ieor: b)
  do i = 1, N
    b = ieor(b, c(i))
  end do

  !$acc parallel loop reduction(.and.: reduction_l)
  do i = 1, N
    reduction_l = d(i) .and. e(i)
  end do

  !$acc parallel loop reduction(.or.: reduction_l)
  do i = 1, N
    reduction_l = d(i) .or. e(i)
  end do

  !$acc parallel loop reduction(.eqv.: reduction_l)
  do i = 1, N
    reduction_l = d(i) .eqv. e(i)
  end do

  !$acc parallel loop reduction(.neqv.: reduction_l)
  do i = 1, N
    reduction_l = d(i) .neqv. e(i)
  end do

  !$acc parallel loop
  do i = 1, N
    if(i == 10) cycle
  end do

end program openacc_parallel_loop_validity
