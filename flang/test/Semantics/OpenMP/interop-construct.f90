! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang %openmp_flags -fopenmp-version=52
! OpenMP Version 5.2
! 14.1 Interop construct
! To check various semantic errors for inteorp construct.

SUBROUTINE test_interop_01()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: obj
  !ERROR: Each interop-var may be specified for at most one action-clause of each INTEROP construct.
  !$OMP INTEROP INIT(TARGETSYNC,TARGET: obj) USE(obj)
  PRINT *, 'pass'
END SUBROUTINE test_interop_01

SUBROUTINE test_interop_02()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: obj
  !ERROR: Each interop-type may be specified at most once.
  !$OMP INTEROP INIT(TARGETSYNC,TARGET,TARGETSYNC: obj)
  PRINT *, 'pass'
END SUBROUTINE test_interop_02

SUBROUTINE test_interop_03()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: obj
  !ERROR: A DEPEND clause can only appear on the directive if the interop-type includes TARGETSYNC
  !$OMP INTEROP INIT(TARGET: obj) DEPEND(INOUT: obj)
  PRINT *, 'pass'
END SUBROUTINE test_interop_03
