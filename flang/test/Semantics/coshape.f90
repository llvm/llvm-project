! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
! Check for semantic errors in coshape() function,
! as defined in section 16.9.55 of the Fortran
! 2018 standard

program coshape_tests
  use iso_c_binding, only : c_int32_t, c_int64_t
  implicit none

  integer array(1), non_coarray(1), scalar_coarray[*], array_coarray(1)[*], non_constant, scalar_result
  real real_coarray[*]
  complex complex_coarray[*]
  character char_array(1)
  logical non_integer, logical_coarray[*]
  integer, allocatable :: codimensions(:)

  !___ standard-conforming statement with no optional arguments present ___
  codimensions = coshape(scalar_coarray)
  codimensions = coshape(array_coarray)
  codimensions = coshape(array_coarray(1))
  codimensions = coshape(scalar_coarray[1])
  codimensions = coshape(real_coarray)
  codimensions = coshape(logical_coarray)
  codimensions = coshape(complex_coarray)
  codimensions = coshape(coarray=scalar_coarray)

  !___ standard-conforming statements with optional kind argument present ___
  codimensions = coshape(scalar_coarray, c_int32_t)
  codimensions = coshape(real_coarray, kind=c_int32_t)
  codimensions = coshape(coarray=logical_coarray, kind=c_int32_t)
  codimensions = coshape(kind=c_int32_t, coarray=complex_coarray)

  !___ non-conforming statements ___
  ! coarray argument must be a coarray
  codimensions = coshape(non_coarray)

  ! kind argument must be an integer
  codimensions = coshape(scalar_coarray, non_integer)

  ! kind argument must be a constant expression
  codimensions = coshape(real_coarray, non_constant)

  ! kind argument must be an integer scalar
  codimensions = coshape(complex_coarray, array)

  ! missing all arguments
  codimensions = coshape()

  ! missing mandatory argument
  codimensions = coshape(kind=c_int32_t)

  ! incorrect typing for mandatory argument
  codimensions = coshape(3.4)

  ! incorrect typing for coarray argument
  codimensions = coshape(coarray=3.4)

  ! too many arguments
  codimensions = coshape(scalar_coarray, c_int32_t, 0)

  ! incorrect typing with correct keyword for coarray argument
  codimensions = coshape(coarray=non_coarray)

  ! correct typing with incorrect keyword for coarray argument
  codimensions = coshape(c=real_coarray)

  ! incorrect typing with correct keyword for kind argument
  codimensions = coshape(complex_coarray, kind=non_integer)

  ! correct typing with incorrect keyword for kind argument
  codimensions = coshape(logical_coarray, kinds=c_int32_t)

  ! repeated keyword for coarray argument
  codimensions = coshape(coarray=scalar_coarray, coarray=real_coarray)

  ! repeated keyword for kind argument
  codimensions = coshape(real_coarray, kind=c_int32_t, kind=c_int64_t)

  ! result must be a rank 1 array
  scalar_result = coshape(scalar_coarray)

  ! result must be an integer array
  char_array = coshape(real_coarray)

end program coshape_tests
