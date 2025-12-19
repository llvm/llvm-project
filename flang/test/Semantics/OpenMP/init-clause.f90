!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=60

subroutine f00
  integer :: x, y
  !ERROR: The INIT clause is not allowed when the DEPOBJ directive has an argument
  !ERROR: The 'depinfo-modifier' modifier is required on a DEPOBJ construct
  !$omp depobj(x) init(y)
end

subroutine f01
  integer :: x, y, z
  !ERROR: DEPOBJ syntax with no argument is not handled yet
  !ERROR: 'DEPOBJ' is not am allowed value of the 'depinfo-modifier' modifier
  !$omp depobj init(depobj(y): z)
end

subroutine f02
  integer :: x, y, z
  !ERROR: DEPOBJ syntax with no argument is not handled yet
  !ERROR: The 'depinfo-modifier' modifier is required on a DEPOBJ construct
  !ERROR: The 'prefer-type' modifier is not allowed on a DEPOBJ construct
  !$omp depobj init(prefer_type({fr("frid")}): z)
end

subroutine f03
  integer :: x, y
  !ERROR: The 'depinfo-modifier' is not allowed on INTEROP construct
  !$omp interop init(mutexinoutset(x): y)
end
