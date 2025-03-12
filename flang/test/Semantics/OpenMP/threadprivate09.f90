! RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags

! Check that namelist variables cannot be used with threadprivate

module m
  integer :: nam1
  common /com1/nam1
  namelist /nam/nam1

  !ERROR: A variable in a THREADPRIVATE directive cannot appear in a NAMELIST
  !$omp threadprivate(/com1/)
end
