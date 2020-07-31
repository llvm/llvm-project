! RUN: %S/test_errors.sh %s %t %f18 -fopenacc

! Check OpenACC restruction in branch in and out of some construct
!

program openacc_clause_validity

  implicit none

  integer :: i
  integer :: N = 256
  real(8) :: a(256)

  !$acc parallel
  !$acc loop
  do i = 1, N
    a(i) = 3.14
    !ERROR: RETURN statement is not allowed in a PARALLEL construct
    return
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop
  do i = 1, N
    a(i) = 3.14
    if(i == N-1) THEN
      !ERROR: EXIT statement is not allowed in a PARALLEL construct
      exit
    end if
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop
  do i = 1, N
    a(i) = 3.14
    if(i == N-1) THEN
      !ERROR: STOP statement is not allowed in a PARALLEL construct
      stop 999
    end if
  end do
  !$acc end parallel

  !$acc kernels
  do i = 1, N
    a(i) = 3.14
    !ERROR: RETURN statement is not allowed in a KERNELS construct
    return
  end do
  !$acc end kernels

  !$acc kernels
  do i = 1, N
    a(i) = 3.14
    if(i == N-1) THEN
      !ERROR: EXIT statement is not allowed in a KERNELS construct
      exit
    end if
  end do
  !$acc end kernels

  !$acc kernels
  do i = 1, N
    a(i) = 3.14
    if(i == N-1) THEN
      !ERROR: STOP statement is not allowed in a KERNELS construct
      stop 999
    end if
  end do
  !$acc end kernels

  !$acc serial
  do i = 1, N
    a(i) = 3.14
    !ERROR: RETURN statement is not allowed in a SERIAL construct
    return
  end do
  !$acc end serial

  !$acc serial
  do i = 1, N
    a(i) = 3.14
    if(i == N-1) THEN
      !ERROR: EXIT statement is not allowed in a SERIAL construct
      exit
    end if
  end do
  !$acc end serial

  !$acc serial
  do i = 1, N
    a(i) = 3.14
    if(i == N-1) THEN
      !ERROR: STOP statement is not allowed in a SERIAL construct
      stop 999
    end if
  end do
  !$acc end serial

end program openacc_clause_validity
