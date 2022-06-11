! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! Check OpenMP clause validity for the following directives:
!     2.10 Device constructs
program main

  real(8) :: arrayA(256), arrayB(256)
  integer :: N

  arrayA = 1.414
  arrayB = 3.14
  N = 256

  !$omp task
  !ERROR: `DISTRIBUTE` region has to be strictly nested inside `TEAMS` region.
  !$omp distribute 
  do i = 1, N
     a = 3.14
  enddo
  !$omp end distribute
  !$omp end task

  !$omp teams
   do i = 1, N
      !ERROR: Only `DISTRIBUTE` or `PARALLEL` regions are allowed to be strictly nested inside `TEAMS` region.
      !$omp task
      do k = 1, N
         a = 3.14
      enddo
      !$omp end task
   enddo
   !$omp end teams

   !$omp teams
   do i = 1, N
      !$omp parallel
      do k = 1, N
         a = 3.14
      enddo
      !$omp end parallel
   enddo
   !$omp end teams

  !$omp parallel
  !ERROR: `DISTRIBUTE` region has to be strictly nested inside `TEAMS` region.
  !$omp distribute 
  do i = 1, N
     a = 3.14
  enddo
  !$omp end distribute
  !$omp end parallel

  !$omp teams
   !ERROR: Only `DISTRIBUTE` or `PARALLEL` regions are allowed to be strictly nested inside `TEAMS` region.
   !$omp target
      !ERROR: `DISTRIBUTE` region has to be strictly nested inside `TEAMS` region.
      !$omp distribute 
      do i = 1, 10
         j = j + 1
      end do
      !$omp end distribute
   !$omp end target
  !$omp end teams

  !$omp teams 
   !$omp parallel
   do k = 1,10
      print *, "hello"
   end do
   !$omp end parallel
   !$omp distribute firstprivate(a)
   do i = 1, 10
      j = j + 1
   end do
   !$omp end distribute
  !$omp end teams

  !$omp teams 
      !ERROR: Only `DISTRIBUTE` or `PARALLEL` regions are allowed to be strictly nested inside `TEAMS` region.
      !$omp task
      do k = 1,10
         print *, "hello"
      end do
      !$omp end task
      !$omp distribute firstprivate(a)
      do i = 1, 10
         j = j + 1
      end do
      !$omp end distribute
  !$omp end teams

  !$omp task 
      !$omp parallel
      do k = 1,10
         print *, "hello"
      end do
      !$omp end parallel
      !ERROR: `DISTRIBUTE` region has to be strictly nested inside `TEAMS` region.
      !$omp distribute firstprivate(a)
      do i = 1, 10
         j = j + 1
      end do
      !$omp end distribute
  !$omp end task
end program main
