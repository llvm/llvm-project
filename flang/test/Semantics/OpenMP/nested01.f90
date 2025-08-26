! RUN: %python %S/../test_errors.py %s %flang -fopenmp

! Check OpenMP 2.17 Nesting of Regions

  N = 1024
  !$omp do
  do i = 1, N
     !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
     !$omp do
     do j = 1, N
        a = 3.14
     enddo
  enddo

  !$omp do
  do i = 1, N
   !$omp target
   do k = 1,N
     !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
     !$omp do
     do j = 1, N
        a = 3.14
     enddo
   enddo
   !$omp end target
  enddo
  
  !$omp do
  do i = 1, N
     !ERROR: A worksharing region may not be closely nested inside a worksharing, explicit task, taskloop, critical, ordered, atomic, or master region
     !$omp scope
     !$omp end scope
  end do
  !$omp end do

  !$omp do
  do i = 1, N
   !$omp parallel
   do k = 1,N
     !$omp do
     do j = 1, N
        a = 3.14
     enddo
   enddo
   !$omp end parallel
  enddo
end
