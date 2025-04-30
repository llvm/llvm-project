! @@name:	declare_target.2f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	success
program my_fib
integer :: N = 8
!$omp declare target(fib)
   !$omp target
      call fib(N)
   !$omp end target
end program
subroutine fib(N)
integer :: N
!$omp declare target
     print*,"hello from fib"
     !...
end subroutine
