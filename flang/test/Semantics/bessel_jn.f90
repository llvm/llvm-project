! RUN: %python %S/test_errors.py %s %flang_fc1
! Test semantic checking for BESSEL_JN and BESSEL_YN intrinsics

program test_bessel
  real :: x, result_scalar
  real :: xarray(10), result_array(10)
  integer :: n, n1, n2

  ! Valid elemental forms
  result_scalar = bessel_jn(n, x)
  result_scalar = bessel_yn(n, x)
  result_array = bessel_jn(n, xarray)
  result_array = bessel_yn(n, xarray)

  ! Valid transformational forms (x must be scalar)
  result_array(n1:n2) = bessel_jn(n1, n2, x)
  result_array(n1:n2) = bessel_yn(n1, n2, x)

  ! Invalid: transformational form with array x
  !ERROR: 'x=' argument has unacceptable rank 1
  result_array(n1:n2) = bessel_jn(n1, n2, xarray)

  !ERROR: 'x=' argument has unacceptable rank 1
  result_array(n1:n2) = bessel_yn(n1, n2, xarray)
end program
