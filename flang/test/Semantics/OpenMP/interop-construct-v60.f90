! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang %openmp_flags -fopenmp-version=60
! OpenMP Version 6.0
! 14.1 Interop construct
! In OpenMP 6.0 the interop-type modifier on an INIT clause is optional, so
! INIT(obj) alone is accepted. A DEPEND clause, however, requires the
! interop-type to include TARGETSYNC, which an untyped INIT does not provide,
! so INIT(obj) combined with DEPEND is diagnosed.

SUBROUTINE test_interop_untyped()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: obj
  !$OMP INTEROP INIT(obj)
  PRINT *, 'pass'
END SUBROUTINE test_interop_untyped

SUBROUTINE test_interop_untyped_depend()
  USE omp_lib
  INTEGER(OMP_INTEROP_KIND) :: obj
  !ERROR: A DEPEND clause can only appear on the directive if the interop-type includes TARGETSYNC
  !$OMP INTEROP INIT(obj) DEPEND(INOUT: obj)
  PRINT *, 'pass'
END SUBROUTINE test_interop_untyped_depend
