! @@name:	declare_target.1f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	success
module module_fib
contains
   subroutine fib(N)
      integer :: N
      !$omp declare target
      !...
   end subroutine
end module
module params
integer :: THRESHOLD=1000000
end module
program my_fib
use params
use module_fib
   !$omp target if( N > THRESHOLD )
      call fib(N)
   !$omp end target
end program
