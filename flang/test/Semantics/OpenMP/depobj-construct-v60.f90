!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine f00
  integer :: obj
  !WARNING: DEPEND clause is no longer allowed on DEPOBJ directive since OpenMP v6.0 [-Wopenmp-deprecated]
  !ERROR: A DEPEND clause on a DEPOBJ construct must not have SINK or SOURCE as dependence type
  !WARNING: 'dependence-type' modifier is no longer allowed on DEPEND clause since OpenMP v5.2 [-Wopenmp-deprecated]
  !$omp depobj(obj) depend(source)
end

subroutine f01
  integer :: obj
  integer :: x, y
  !WARNING: DEPEND clause is no longer allowed on DEPOBJ directive since OpenMP v6.0 [-Wopenmp-deprecated]
  !ERROR: A DEPEND clause on a DEPOBJ construct must only specify one locator
  !$omp depobj(obj) depend(in: x, y)
end

subroutine f02
  integer :: obj
  integer :: x(10)
  !WARNING: DEPEND clause is no longer allowed on DEPOBJ directive since OpenMP v6.0 [-Wopenmp-deprecated]
  !WARNING: An iterator-modifier may specify multiple locators, a DEPEND clause on a DEPOBJ construct must only specify one locator
  !$omp depobj(obj) depend(iterator(i = 1:10), in: x(i))
end

subroutine f03
  integer :: obj, jbo
  !ERROR: The DESTROY clause must refer to the same object as the DEPOBJ construct
  !$omp depobj(obj) destroy(jbo)
end

subroutine f04
  integer :: obj1, obj2
  !ERROR: When DEPOBJ directive is specified with an argument only one argument may be present
  !WARNING: The DESTROY clause without argument on DEPOBJ construct is deprecated in OpenMP v5.2
  !$omp depobj(ob1, obj2) destroy
end

subroutine f05
  !ERROR: DEPOBJ syntax with no argument is not handled yet
  !$omp depobj update(in)
end

subroutine f06
  integer :: obj
  !ERROR: When DEPOBJ directive is specified with an argument a single clause must be present
  !WARNING: The DESTROY clause without argument on DEPOBJ construct is deprecated in OpenMP v5.2
  !$omp depobj(obj) update(in) destroy
end

subroutine f07
  integer :: obj1, obj2
  !ERROR: DEPOBJ syntax with no argument is not handled yet
  !$omp depobj destroy(obj1) destroy(obj2)
end
