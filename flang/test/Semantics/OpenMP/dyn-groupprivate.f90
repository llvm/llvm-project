!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=61

subroutine f00(n)
  implicit none
  integer :: n
  !Expect no diagnostic
  !$omp target dyn_groupprivate(n)
  !$omp end target
end

subroutine f01(n)
  implicit none
  integer :: n
  !Expect no diagnostic
  !$omp target dyn_groupprivate(strict: n)
  !$omp end target
end

subroutine f02(n)
  implicit none
  integer :: n
  !Expect no diagnostic
  !$omp target dyn_groupprivate(fallback, cgroup: n)
  !$omp end target
end

subroutine f03(n)
  implicit none
  integer :: n
  !If absent, access-group is assumed to be CGROUP
  !ERROR: The access-group modifier can only occur on a single clause in a construct
  !$omp target dyn_groupprivate(n) dyn_groupprivate(cgroup: n)
  !$omp end target
end

subroutine f04(n, m)
  implicit none
  integer :: n, m
  !$omp target dyn_groupprivate(fallback, cgroup: n) &
  !ERROR: The access-group modifier can only occur on a single clause in a construct
  !$omp &      dyn_groupprivate(cgroup: m)
  !$omp end target
end

