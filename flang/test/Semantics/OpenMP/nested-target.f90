! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp -Werror -pedantic

! OpenMP Version 5.0
! Check OpenMP construct validity for the following directives:
! 2.12.5 Target Construct

program main
  integer :: i, j, N = 10, n1, n2, res(100)
  real :: a, arrayA(512), arrayB(512), ai(10)
  real, allocatable :: B(:)

  !$omp target
  !PORTABILITY: If TARGET UPDATE directive is nested inside TARGET region, the behaviour is unspecified
  !$omp target update from(arrayA) to(arrayB)
  do i = 1, 512
    arrayA(i) = arrayB(i)
  end do
  !$omp end target

  !$omp parallel
  !$omp target
  !$omp parallel
  !PORTABILITY: If TARGET UPDATE directive is nested inside TARGET region, the behaviour is unspecified
  !$omp target update from(arrayA) to(arrayB)
  do i = 1, 512
    arrayA(i) = arrayB(i)
  end do
  !$omp end parallel
  !$omp end target
  !$omp end parallel

  !$omp target
  !PORTABILITY: If TARGET DATA directive is nested inside TARGET region, the behaviour is unspecified
  !$omp target data map(to: a)
  do i = 1, N
    a = 3.14
  end do
  !$omp end target data
  !$omp end target

  allocate(B(N))
  !$omp target
  !PORTABILITY: If TARGET ENTER DATA directive is nested inside TARGET region, the behaviour is unspecified
  !$omp target enter data map(alloc:B)
  !$omp end target

  !$omp target
  !PORTABILITY: If TARGET EXIT DATA directive is nested inside TARGET region, the behaviour is unspecified
  !$omp target exit data map(delete:B)
  !$omp end target
  deallocate(B)

  n1 = 10
  n2 = 10
  !$omp target teams map(to:a)
  !PORTABILITY: If TARGET DATA directive is nested inside TARGET region, the behaviour is unspecified
  !$omp target data map(n1,n2)
  do i=1, n1
     do j=1, n2
      res((i-1)*10+j) = i*j
     end do
  end do
  !$omp end target data
  !$omp end target teams

  !$omp target teams map(to:a) map(from:n1,n2)
  !PORTABILITY: If TARGET TEAMS DISTRIBUTE PARALLEL DO directive is nested inside TARGET region, the behaviour is unspecified
  !$omp target teams distribute parallel do
  do i=1, n1
     do j=1, n2
      res((i-1)*10+j) = i*j
     end do
  end do
  !$omp end target teams distribute parallel do
  !$omp end target teams

end program main
