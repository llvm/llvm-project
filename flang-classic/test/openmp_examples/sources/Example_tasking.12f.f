! @@name:	tasking.12f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	rt-error
subroutine foo()
  integer :: x
  x = 2
!$omp task mergeable
  x = x + 1
!$omp end task
!$omp taskwait
  print *, x   ! prints 2 or 3
end subroutine
