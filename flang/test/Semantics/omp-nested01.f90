! RUN: %S/test_errors.sh %s %t %f18 -fopenmp

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
end
