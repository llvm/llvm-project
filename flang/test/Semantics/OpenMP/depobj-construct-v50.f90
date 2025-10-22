!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=50

subroutine f00
  integer :: obj
!ERROR: A DEPEND clause on a DEPOBJ construct must not have SINK, SOURCE or DEPOBJ as dependence type
  !$omp depobj(obj) depend(source)
end

subroutine f01
  integer :: obj
  integer :: x, y
!ERROR: A DEPEND clause on a DEPOBJ construct must only specify one locator
  !$omp depobj(obj) depend(in: x, y)
end

subroutine f02
  integer :: obj
  integer :: x(10)
!WARNING: An iterator-modifier may specify multiple locators, a DEPEND clause on a DEPOBJ construct must only specify one locator
  !$omp depobj(obj) depend(iterator(i = 1:10), in: x(i))
end

subroutine f03
  integer :: obj, jbo
!ERROR: The DESTROY clause must refer to the same object as the DEPOBJ construct
!WARNING: The object parameter in DESTROY clause on DEPOPJ construct is not allowed in OpenMP v5.0, try -fopenmp-version=52
  !$omp depobj(obj) destroy(jbo)
end

subroutine f04
  integer :: obj1, obj2
!ERROR: The DEPOBJ directive requires a single argument
  !$omp depobj(ob1, obj2) destroy
end

subroutine f05
!ERROR: The DEPOBJ directive requires a single argument
  !$omp depobj update(in)
end

subroutine f06
  integer :: obj
!ERROR: The DEPOBJ construct requires a single clause
  !$omp depobj(obj) update(in) destroy
end
