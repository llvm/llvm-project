!RUN: %python %S/../test_errors.py %s %flang -fopenmp -fopenmp-version=52 -Werror

subroutine f00(x)
  integer :: x
!WARNING: 'self-modifier' modifier is not supported in OpenMP v5.2, try -fopenmp-version=60
  !$omp target map(self: x)
  x = x + 1
  !$omp end target
end

subroutine f01(x)
  integer, pointer :: x
!WARNING: 'ref-modifier' modifier is not supported in OpenMP v5.2, try -fopenmp-version=60
  !$omp target map(ref_ptr: x)
  x = x + 1
  !$omp end target
end

subroutine f02(x)
  integer, pointer :: x
!WARNING: 'ref-modifier' modifier is not supported in OpenMP v5.2, try -fopenmp-version=60
  !$omp target map(ref_ptee: x)
  x = x + 1
  !$omp end target
end

subroutine f03(x)
  integer, pointer :: x
!WARNING: 'ref-modifier' modifier is not supported in OpenMP v5.2, try -fopenmp-version=60
  !$omp target map(ref_ptr_ptee: x)
  x = x + 1
  !$omp end target
end


