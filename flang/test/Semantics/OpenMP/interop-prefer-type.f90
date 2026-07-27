! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang %openmp_flags -fopenmp-version=60
! OpenMP Version 6.0
! 14.1 Interop construct
! The foreign-runtime-identifier in a `prefer_type` modifier must be a constant
! expression of integer OpenMP type or a base language string literal.

SUBROUTINE test_prefer_type_nonconstant()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: obj
  INTEGER :: n
  !ERROR: The foreign-runtime-identifier in a `prefer_type` modifier must be a constant expression of integer OpenMP type or a base language string literal
  !$OMP INTEROP INIT(PREFER_TYPE(n), TARGET: obj)
  PRINT *, 'pass'
END SUBROUTINE test_prefer_type_nonconstant

SUBROUTINE test_prefer_type_constant_int()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: obj
  !$OMP INTEROP INIT(PREFER_TYPE(omp_ifr_cuda), TARGET: obj)
  PRINT *, 'pass'
END SUBROUTINE test_prefer_type_constant_int

SUBROUTINE test_prefer_type_string()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: obj
  !$OMP INTEROP INIT(PREFER_TYPE("cuda"), TARGET: obj)
  PRINT *, 'pass'
END SUBROUTINE test_prefer_type_string
