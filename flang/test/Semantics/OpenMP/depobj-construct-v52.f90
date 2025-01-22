!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52

subroutine f00
  integer :: obj
!WARNING: SOURCE dependence type is deprecated in OpenMP v5.2
!ERROR: A DEPEND clause on a DEPOBJ construct must not have SINK or SOURCE as dependence type
  !$omp depobj(obj) depend(source)
end

subroutine f03
  integer :: obj, jbo
!Note: no portability message
!ERROR: The DESTROY clause must refer to the same object as the DEPOBJ construct
  !$omp depobj(obj) destroy(jbo)
end

subroutine f06
  integer :: obj
!WARNING: The DESTROY clause without argument on DEPOBJ construct is deprecated in OpenMP v5.2
  !$omp depobj(obj) destroy
end
