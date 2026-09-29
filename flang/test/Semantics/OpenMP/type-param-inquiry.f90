!RUN: %python %S/../test_errors.py %s %flang_fc1 %openmp_flags -fopenmp-version=60

subroutine f00
  integer :: x
!ERROR: A type parameter inquiry cannot appear on the INIT clause
  !$omp interop init(x%kind)
end

subroutine f01
  integer :: x
!ERROR: A type parameter inquiry cannot appear on the USE clause
  !$omp interop use(x%kind)
end

subroutine f02
  integer :: x
!ERROR: A type parameter inquiry cannot appear on the DESTROY clause
  !$omp interop destroy(x%kind)
end

subroutine f03
  integer :: x
!ERROR: A type parameter inquiry cannot appear on the DETACH clause
  !$omp task detach(x%kind)
  !$omp end task
end

subroutine f04
  integer :: x
!ERROR: DEPOBJ syntax with no argument is not handled yet
!ERROR: A type parameter inquiry cannot appear on the INIT clause
!ERROR: The 'depinfo-modifier' modifier is required on a DEPOBJ construct
  !$omp depobj init(x%kind)
end

subroutine f05
  integer :: x
!ERROR: DEPOBJ syntax with no argument is not handled yet
!ERROR: A type parameter inquiry cannot appear on the DESTROY clause
  !$omp depobj destroy(x%kind)
end
