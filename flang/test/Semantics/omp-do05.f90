! RUN: %S/test_errors.sh %s %t %flang -fopenmp
! REQUIRES: shell
! OpenMP Version 4.5
! 2.7.1 Loop Construct restrictions on single directive.


program omp_do

  integer n
  integer i,j,k
  !$omp do
  do i=1,10
    !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
    !$omp single
    do j=1,10
      print *,"hello"
    end do
    !$omp end single
  end do
  !$omp end do

  !$omp parallel do
  do i=1,10
    !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
    !$omp single
    do j=1,10
      print *,"hello"
    end do
    !$omp end single
  end do
  !$omp end parallel do

  !$omp parallel do simd
  do i=1,10
    !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
    !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
    !$omp single
    do j=1,10
      print *,"hello"
    end do
    !$omp end single
  end do
  !$omp end parallel do simd

  !ERROR: `DISTRIBUTE` region has to be strictly nested inside `TEAMS` region.
  !$omp distribute parallel do
  do i=1,10
    !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
    !$omp single
    do j=1,10
      print *,"hello"
    end do
    !$omp end single
  end do
  !$omp end distribute parallel do

  !ERROR: `DISTRIBUTE` region has to be strictly nested inside `TEAMS` region.
  !$omp distribute parallel do simd
  do i=1,10
    !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
    !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
    !$omp single
    do j=1,10
      print *,"hello"
    end do
    !$omp end single
  end do
  !$omp end distribute parallel do simd

  !$omp target parallel do 
  do i=1,10
    !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
    !$omp single
    do j=1,10
      print *,"hello"
    end do
    !$omp end single
  end do
  !$omp end target parallel do

  !$omp target parallel do simd
  do i=1,10
    !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
    !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
    !$omp single
    do j=1,10
      print *,"hello"
    end do
    !$omp end single
  end do
  !$omp end target parallel do simd

  !$omp target teams distribute parallel do
  do i=1,10
    !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
    !$omp single
    do j=1,10
      print *,"hello"
    end do
    !$omp end single
  end do
  !$omp end target teams distribute parallel do

  !$omp target teams distribute parallel do simd
  do i=1,10
    !ERROR: The only OpenMP constructs that can be encountered during execution of a 'SIMD' region are the `ATOMIC` construct, the `LOOP` construct, the `SIMD` construct and the `ORDERED` construct with the `SIMD` clause.
    !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
    !$omp single
    do j=1,10
      print *,"hello"
    end do
    !$omp end single
  end do
  !$omp end target teams distribute parallel do simd

  !$omp do
  do i=1,10
    !$omp task
    do j=1,10
      !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
      !$omp single
      do k=1,10
        print *,"hello"
      end do
      !$omp end single
    end do
    !$omp end task
  end do
  !$omp end do

  !$omp do
  do i=1,10
    !$omp parallel
    do j=1,10
      !$omp single
      do k=1,10
        print *,"hello"
      end do
      !$omp end single
    end do
    !$omp end parallel
  end do
  !$omp end do

!$omp do
do i=1,10
  !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
  !$omp single
  do j=1,10
    !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
    !$omp single
    do k=1,10
      print *,"hello"
    end do
    !$omp end single
  end do
  !$omp end single

  !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
  !$omp single
    do k=1,10
      print *,"hello"
    end do
  !$omp end single

  !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
  !$omp do
    do k=1,10
      print *,"hello"
    end do
  !$omp end do
end do
!$omp end do

  !$omp parallel default(shared)
  !$omp do
  do i = 1, n
    !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
    !$omp single
    call work(i, 1)
    !$omp end single
  end do
  !$omp end do
  !$omp end parallel

  !$omp parallel default(shared)
  !$omp do
  do i = 1, n
    !$omp task
    do j=1,10
      !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
      !$omp single
      call work(i, 1)
      !$omp end single
    end do
    !$omp end task
  end do
  !$omp end do
  !$omp end parallel

end program omp_do
