! @@name:	teams.4f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
module arrays
integer,parameter :: N=1024*1024
real :: B(N), C(N)
end module
function dotprod() result(sum)
use arrays
   real    :: sum
   integer :: i
   sum = 0.0e0
   !$omp target map(to: B, C)
   !$omp teams num_teams(8) thread_limit(16)
   !$omp distribute parallel do reduction(+:sum) &
   !$omp&  dist_schedule(static, 1024) schedule(static, 64)
      do i = 1,N
         sum = sum + B(i) * C(i)
      end do
   !$omp end teams
   !$omp end target
end function
