! RUN: %S/../test_errors.sh %s %t %flang -fopenacc

! Check OpenACC canonalization validity for the construct defined below:
!   2.9 Loop
!   2.11 Parallel Loop
!   2.11 Kernels Loop
!   2.11 Serial Loop

program openacc_clause_validity

  implicit none

  integer :: i, j
  integer :: N = 256
  real(8) :: a(256)
  real(8) :: aa(256, 256)

  !ERROR: A DO loop must follow the LOOP directive
  !$acc loop
  i = 1

  !ERROR: DO loop after the LOOP directive must have loop control
  !$acc loop
  do
  end do

  !ERROR: A DO loop must follow the PARALLEL LOOP directive
  !$acc parallel loop
  i = 1

  !ERROR: A DO loop must follow the KERNELS LOOP directive
  !$acc kernels loop
  i = 1

  !ERROR: A DO loop must follow the SERIAL LOOP directive
  !$acc serial loop
  i = 1

  !ERROR: The END PARALLEL LOOP directive must follow the DO loop associated with the loop construct
  !$acc end parallel loop

  !ERROR: The END KERNELS LOOP directive must follow the DO loop associated with the loop construct
  !$acc end kernels loop

  !ERROR: The END SERIAL LOOP directive must follow the DO loop associated with the loop construct
  !$acc end serial loop

  !$acc parallel loop
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc kernels loop
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc serial loop
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc parallel loop
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel loop

  !$acc kernels loop
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end kernels loop

  !$acc serial loop
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end serial loop

  !ERROR: DO loop after the PARALLEL LOOP directive must have loop control
  !$acc parallel loop
  do
  end do

  !ERROR: DO loop after the KERNELS LOOP directive must have loop control
  !$acc kernels loop
  do
  end do

  !ERROR: DO loop after the SERIAL LOOP directive must have loop control
  !$acc serial loop
  do
  end do

  !$acc parallel
  !ERROR: The loop construct with the TILE clause must be followed by 2 tightly-nested loops
  !$acc loop tile(2, 2)
  do i = 1, N
    a(i) = 3.14
  end do
  !$acc end parallel

  !ERROR: The loop construct with the TILE clause must be followed by 2 tightly-nested loops
  !$acc parallel loop tile(2, 2)
  do i = 1, N
    a(i) = 3.14
  end do

  !$acc parallel
  !ERROR: TILE and COLLAPSE clause may not appear on loop construct associated with DO CONCURRENT
  !$acc loop collapse(2)
  do concurrent (i = 1:N, j = 1:N)
    aa(i, j) = 3.14
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: TILE and COLLAPSE clause may not appear on loop construct associated with DO CONCURRENT
  !$acc loop tile(2, 2)
  do concurrent (i = 1:N, j = 1:N)
    aa(i, j) = 3.14
  end do
  !$acc end parallel

end program openacc_clause_validity
