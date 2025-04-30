! @@name:	declare_target.3f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
module my_arrays
!$omp declare target (N, p, v1, v2)
integer, parameter :: N=1000
real               :: p(N), v1(N), v2(N)
end module
subroutine vec_mult()
use my_arrays
   integer :: i
   call init(v1, v2, N);
   !$omp target update to(v1, v2)
   !$omp target
   !$omp parallel do
   do i = 1,N
     p(i) = v1(i) * v2(i)
   end do
   !$omp end target
   !$omp target update from (p)
   call output(p, N)
end subroutine
