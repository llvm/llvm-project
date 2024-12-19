! REQUIRES: openmp_runtime

! RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52
! OpenMP Version 5.2
! 14.1 Interop construct
! To check various semantic errors for inteorp construct.

subroutine test_interop_01()
  use omp_lib
  integer(omp_interop_kind) :: obj
  !ERROR: Each interop-var may be speciﬁed for at most one action-clause of each interop construct.
  !$omp interop init(targetsync,target: obj) use(obj)
  print *, 'pass'
end subroutine test_interop_01

subroutine test_interop_02()
  use omp_lib
  integer(omp_interop_kind) :: obj
  !ERROR: Each interop-type may be speciﬁed at most once.
  !$omp interop init(targetsync,target,targetsync: obj)
  print *, 'pass'
end subroutine test_interop_02

subroutine test_interop_03()
  use omp_lib
  integer(omp_interop_kind) :: obj
  !ERROR: A depend clause can only appear on the directive if the interop-type includes targetsync
  !$omp interop init(target: obj) depend(inout: obj)
  print *, 'pass'
end subroutine test_interop_03