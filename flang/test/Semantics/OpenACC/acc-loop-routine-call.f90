! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Semantic checks for calling !$acc routine procedures from LOOP bodies

module acc_loop_routine_call_m
  implicit none
contains

  subroutine r_seq()
    !$acc routine seq
  end subroutine r_seq

  subroutine r_vector()
    !$acc routine vector
  end subroutine r_vector

  subroutine r_worker()
    !$acc routine worker
  end subroutine r_worker

  subroutine r_gang()
    !$acc routine gang
  end subroutine r_gang

  subroutine r_gang_dim1()
    !$acc routine gang(dim:1)
  end subroutine r_gang_dim1

  subroutine r_gang_dim2()
    !$acc routine gang(dim:2)
  end subroutine r_gang_dim2

  subroutine r_gang_dim3()
    !$acc routine gang(dim:3)
  end subroutine r_gang_dim3

end module acc_loop_routine_call_m

program acc_loop_routine_call
  use acc_loop_routine_call_m
  implicit none
  integer, parameter :: n = 8
  integer :: i

  !$acc parallel
  !$acc loop vector
  do i = 1, n
    call r_seq()
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Calling GANG routine inside VECTOR loop is not allowed
  !$acc loop vector
  do i = 1, n
    call r_gang()
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Calling GANG(1) routine inside VECTOR loop is not allowed
  !$acc loop vector
  do i = 1, n
    call r_gang_dim1()
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Calling WORKER routine inside VECTOR loop is not allowed
  !$acc loop vector
  do i = 1, n
    call r_worker()
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Calling VECTOR routine inside VECTOR loop is not allowed
  !$acc loop vector
  do i = 1, n
    call r_vector()
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Calling GANG(3) routine inside VECTOR loop is not allowed
  !$acc loop vector
  do i = 1, n
    call r_gang_dim3()
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop worker
  do i = 1, n
    call r_seq()
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop worker
  do i = 1, n
    call r_vector()
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Calling GANG routine inside WORKER loop is not allowed
  !$acc loop worker
  do i = 1, n
    call r_gang()
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Calling GANG(1) routine inside WORKER loop is not allowed
  !$acc loop worker
  do i = 1, n
    call r_gang_dim1()
  end do
  !$acc end parallel
  
  !$acc parallel
  !ERROR: Calling GANG(2) routine inside WORKER loop is not allowed
  !$acc loop worker
  do i = 1, n
    call r_gang_dim2()
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Calling GANG(3) routine inside WORKER loop is not allowed
  !$acc loop worker
  do i = 1, n
    call r_gang_dim3()
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Calling WORKER routine inside WORKER loop is not allowed
  !$acc loop worker
  do i = 1, n
    call r_worker()
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop gang(dim:2)
  do i = 1, n
    call r_seq()
    call r_vector()
    call r_gang_dim1()
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Calling GANG(1) routine inside GANG loop is not allowed
  !$acc loop gang
  do i = 1, n
    call r_gang_dim1()
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Calling GANG routine inside GANG loop is not allowed
  !$acc loop gang
  do i = 1, n
    call r_gang()
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop gang(dim:2)
  do i = 1, n
    call r_gang_dim1()
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Calling GANG(3) routine inside GANG(2) loop is not allowed
  !$acc loop gang(dim:2)
  do i = 1, n
    call r_gang_dim3()
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Calling GANG(2) routine inside GANG(2) loop is not allowed
  !$acc loop gang(dim:2)
  do i = 1, n
    call r_gang_dim2()
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop gang(dim:3)
  do i = 1, n
    call r_gang_dim1()
    call r_gang_dim2()
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Calling GANG(3) routine inside GANG(3) loop is not allowed
  !$acc loop gang(dim:3)
  do i = 1, n
    call r_gang_dim3()
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Calling GANG(3) routine inside GANG loop is not allowed
  !$acc loop gang
  do i = 1, n
    call r_gang_dim3()
  end do
  !$acc end parallel

end program acc_loop_routine_call
