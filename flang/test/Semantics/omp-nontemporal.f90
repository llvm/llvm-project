! RUN: %python %S/test_errors.py %s  %flang -fopenmp
! REQUIRES: shell
! Check OpenMP clause validity for NONTEMPORAL clause

program omp_simd
  integer i
  integer, allocatable :: a(:)

  allocate(a(10))

  !$omp simd nontemporal(a)
  do i = 1, 10
    a(i) = i
  end do
  !$omp end simd

  !$omp parallel do simd nontemporal(a)
  do i = 1, 10
    a(i) = i
  end do
  !$omp end parallel do simd
 
  !$omp parallel do simd nontemporal(a)
  do i = 1, 10
    a(i) = i
  end do
  !$omp end parallel do simd

  !ERROR: NONTEMPORAL clause is not allowed on the DO SIMD directive
  !$omp do simd nontemporal(a)
  do i = 1, 10
    a(i) = i
  end do
  !$omp end do simd

  !$omp taskloop simd nontemporal(a)
  do i = 1, 10
    a(i) = i
  end do
  !$omp end taskloop simd

  !$omp teams
  !$omp distribute parallel do simd nontemporal(a)
  do i = 1, 10
    a(i) = i
  end do
  !$omp end distribute parallel do simd
  !$omp end teams

  !$omp teams
  !$omp distribute simd nontemporal(a)
  do i = 1, 10
    a(i) = i
  end do
  !$omp end distribute simd
  !$omp end teams

  !$omp target parallel do simd nontemporal(a)
  do i = 1, 10
    a(i) = i
  end do
  !$omp end target parallel do simd

  !$omp target simd nontemporal(a)
  do i = 1, 10
    a(i) = i
  end do
  !$omp end target simd

  !$omp teams distribute simd nontemporal(a)
  do i = 1, 10
    a(i) = i
  end do
  !$omp end teams distribute simd

  !$omp teams distribute parallel do simd nontemporal(a)
  do i = 1, 10
    a(i) = i
  end do
  !$omp end teams distribute parallel do simd

  !$omp target teams distribute parallel do simd nontemporal(a)
  do i = 1, 10
    a(i) = i
  end do
  !$omp end target teams distribute parallel do simd

  !$omp target teams distribute simd nontemporal(a)
  do i = 1, 10
    a(i) = i
  end do
  !$omp end target teams distribute simd

  
end program omp_simd
