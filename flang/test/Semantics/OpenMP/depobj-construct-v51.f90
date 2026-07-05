!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=51

subroutine f04
  integer :: obj
!ERROR: An UPDATE clause on a DEPOBJ construct must not have SINK, SOURCE or DEPOBJ as dependence type
  !$omp depobj(obj) update(source)
end

subroutine f05
  integer :: obj
!ERROR: An UPDATE clause on a DEPOBJ construct must not have SINK, SOURCE or DEPOBJ as dependence type
  !$omp depobj(obj) update(depobj)
end
