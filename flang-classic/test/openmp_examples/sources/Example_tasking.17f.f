! @@name:	tasking.17f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	success
program example
   integer :: x
   !$omp parallel
   !$omp single
      !$omp task shared(x) depend(out: x)
         x = 1
      !$omp end task
      !$omp task shared(x) depend(out: x)
         x = 2
      !$omp end task
      !$omp taskwait
      print*, "x = ", x
   !$omp end single
   !$omp end parallel
end program
