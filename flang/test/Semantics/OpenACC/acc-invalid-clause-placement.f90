! RUN: %python %S/../test_errors.py %s %flang -fopenacc

subroutine data_clauses_on_loop(a, p)
  real :: a(10)
  real, pointer :: p
  integer :: i

  !ERROR: COPY clause is not allowed on the LOOP directive
  !ERROR: COPYIN clause is not allowed on the LOOP directive
  !ERROR: COPYOUT clause is not allowed on the LOOP directive
  !ERROR: CREATE clause is not allowed on the LOOP directive
  !ERROR: NO_CREATE clause is not allowed on the LOOP directive
  !ERROR: PRESENT clause is not allowed on the LOOP directive
  !$acc loop copy(a) copyin(a) copyout(a) create(a) no_create(a) present(a)
  do i = 1, 10
    a(i) = real(i)
  end do

  !ERROR: DEVICEPTR clause is not allowed on the LOOP directive
  !ERROR: ATTACH clause is not allowed on the LOOP directive
  !ERROR: DELETE clause is not allowed on the LOOP directive
  !ERROR: DETACH clause is not allowed on the LOOP directive
  !$acc loop deviceptr(p) attach(p) delete(a) detach(p)
  do i = 1, 10
    a(i) = real(i)
  end do
end subroutine

subroutine reduction_on_kernels(x)
  integer :: x

  !ERROR: REDUCTION clause is not allowed on the KERNELS directive
  !$acc kernels reduction(+:x)
  x = x + 1
  !$acc end kernels
end subroutine
