! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

! OpenMP Version 5.0
! Check OpenMP construct validity for the following directives:
! 2.12.5 Target Construct

program main
  integer :: i, j, N = 10
  real :: a, arrayA(512), arrayB(512), ai(10)
  real, allocatable :: B(:)

  !$omp target
  !WARNING: If TARGET UPDATE directive is nested inside TARGET region, the behaviour is unspecified
  !$omp target update from(arrayA) to(arrayB)
  do i = 1, 512
    arrayA(i) = arrayB(i)
  end do
  !$omp end target

  !$omp parallel
  !$omp target
  !$omp parallel
  !WARNING: If TARGET UPDATE directive is nested inside TARGET region, the behaviour is unspecified
  !$omp target update from(arrayA) to(arrayB)
  do i = 1, 512
    arrayA(i) = arrayB(i)
  end do
  !$omp end parallel
  !$omp end target
  !$omp end parallel

  !$omp target
  !WARNING: If TARGET DATA directive is nested inside TARGET region, the behaviour is unspecified
  !$omp target data map(to: a)
  do i = 1, N
    a = 3.14
  end do
  !$omp end target data
  !$omp end target

  allocate(B(N))
  !$omp target
  !WARNING: If TARGET ENTER DATA directive is nested inside TARGET region, the behaviour is unspecified
  !$omp target enter data map(alloc:B)
  !$omp end target

  !$omp target
  !WARNING: If TARGET EXIT DATA directive is nested inside TARGET region, the behaviour is unspecified
  !$omp target exit data map(delete:B)
  !$omp end target
  deallocate(B)

end program main
