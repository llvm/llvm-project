! RUN: %python %S/test_errors.py %s %flang_fc1
! Test intrinsic vs non_intrinsic module coexistence
module iso_fortran_env
  integer, parameter :: user_defined_123 = 123
end module
module m1
  use, intrinsic :: iso_fortran_env, only: int32
  !PORTABILITY: Should not USE the non-intrinsic module 'iso_fortran_env' in the same scope as a USE of the intrinsic module
  use, non_intrinsic :: iso_fortran_env, only: user_defined_123
end module
module m2
  use, intrinsic :: iso_fortran_env, only: int32
end module
module m3
  use, non_intrinsic :: iso_fortran_env, only: user_defined_123
end module
module m4
  use :: iso_fortran_env, only: user_defined_123
end module
module m5
  !ERROR: Cannot read module file for module 'ieee_arithmetic': Source file 'ieee_arithmetic.mod' was not found
  use, non_intrinsic :: ieee_arithmetic, only: ieee_selected_real_kind
end module
module notAnIntrinsicModule
end module
module m6
  !ERROR: Cannot read module file for module 'notanintrinsicmodule': Source file 'notanintrinsicmodule.mod' was not found
  use, intrinsic :: notAnIntrinsicModule
end module

