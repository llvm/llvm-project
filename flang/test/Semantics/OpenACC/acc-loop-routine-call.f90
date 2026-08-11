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

  integer function f_seq(x)
    integer, intent(in) :: x
    !$acc routine seq
    f_seq = x
  end function f_seq

  integer function f_vector(x)
    integer, intent(in) :: x
    !$acc routine vector
    f_vector = x
  end function f_vector

  integer function f_worker(x)
    integer, intent(in) :: x
    !$acc routine worker
    f_worker = x
  end function f_worker

  integer function f_gang(x)
    integer, intent(in) :: x
    !$acc routine gang
    f_gang = x
  end function f_gang

  integer function f_gang_dim3(x)
    integer, intent(in) :: x
    !$acc routine gang(dim:3)
    f_gang_dim3 = x
  end function f_gang_dim3

end module acc_loop_routine_call_m

subroutine case4_routine_dim_arg()
  use acc_loop_routine_call_m
  !ERROR: Calling GANG routine inside GANG loop is not allowed
  !$acc routine gang(dim: f_gang(1))
end subroutine case4_routine_dim_arg

program acc_loop_routine_call
  use acc_loop_routine_call_m
  implicit none
  integer, parameter :: n = 8
  integer :: i, j
  real :: a(8)

  a = 0.0

  !$acc parallel
  !$acc loop vector
  do i = 1, n
    call r_seq()
  end do
  !$acc end parallel

  !$acc parallel
  !$acc loop vector
  do i = 1, n
    j = f_seq(i)
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
  !ERROR: Calling VECTOR routine inside VECTOR loop is not allowed
  !$acc loop vector
  do i = 1, n
    j = f_vector(i)
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Calling WORKER routine inside VECTOR loop is not allowed
  !$acc loop vector
  do i = 1, n
    j = f_worker(i)
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Calling GANG routine inside VECTOR loop is not allowed
  !$acc loop vector
  do i = 1, n
    j = f_gang(i)
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
  !$acc loop worker
  do i = 1, n
    j = f_vector(i)
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
  !ERROR: Calling WORKER routine inside WORKER loop is not allowed
  !$acc loop worker
  do i = 1, n
    j = f_worker(i)
  end do
  !$acc end parallel

  !$acc parallel
  !ERROR: Calling GANG routine inside WORKER loop is not allowed
  !$acc loop worker
  do i = 1, n
    j = f_gang(i)
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
  !ERROR: Calling GANG(3) routine inside GANG(2) loop is not allowed
  !$acc loop gang(dim:2)
  do i = 1, n
    j = f_gang_dim3(i)
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

  ! Check firing outside loop-body execution:
  !$acc kernels
  !ERROR: Calling GANG routine inside VECTOR loop is not allowed
  !$acc loop vector(f_gang(n))
  do i = 1, n
    j = i
  end do
  !$acc end kernels

  !$acc parallel
  !ERROR: Calling GANG routine inside VECTOR loop is not allowed
  !$acc loop vector
  do i = 1, f_gang(n)
    j = i
  end do
  !$acc end parallel

  !ERROR: Calling GANG routine inside VECTOR loop is not allowed
  !$acc parallel loop vector if(f_gang(n) > 0)
  do i = 1, n
    a(i) = a(i) + 1.0
  end do

end program acc_loop_routine_call
