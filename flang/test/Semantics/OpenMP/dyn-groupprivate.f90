!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=61

subroutine f00(x)
  integer :: x
  !ERROR: The access-group modifier can only occur on a single clause in a construct
  !$omp target dyn_groupprivate(cgroup: x), dyn_groupprivate(10)
  !$omp end target
end
