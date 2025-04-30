! @@name:	tasking.11f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
subroutine foo()
  integer :: x
  x = 2
!$omp task shared(x) mergeable
  x = x + 1
!$omp end task
!$omp taskwait
  print *, x     ! prints 3
end subroutine
