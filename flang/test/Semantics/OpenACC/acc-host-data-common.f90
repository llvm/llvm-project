! RUN: %python %S/../test_errors.py %s %flang -fopenacc

! Regression test: a variable that lives in a COMMON block must not be
! rejected when it appears in an OpenACC `host_data use_device(...)`
! clause. The CUDA Fortran restriction "object with ATTRIBUTES(USEDEVICE)
! may not be in COMMON" applies only to user-spelled
! `attributes(usedevice)` dummy arguments, not to the internal
! CUDADataAttr::UseDevice marker that name resolution attaches to
! construct-scoped copies of `use_device` operands.

subroutine vadd(a, b, c, n)
  real(8) :: a(*), b(*), c(*)
  integer :: n, i
  do i = 1, n - 1
    c(i) = a(i) + b(i)
  end do
end subroutine

program acc_host_data_common
  integer, parameter :: N = 100
  real(8) :: a(2:N), b(2:N), c0(2:N), c1(2:N)
  common /arrays/ a, b, c0, c1
  integer :: i

  !$acc data copy(a, b, c0)
  !$acc parallel loop
  do i = 2, N
    a(i) = i
    b(i) = 2.0_8 * i
  end do

  !$acc host_data use_device(a, b, c0)
  call vadd(a, b, c0, N)
  !$acc end host_data
  !$acc end data
end program
