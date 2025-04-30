! @@name:	declare_target.5f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
module my_global_array
!$omp declare target (N,Q)
integer, parameter :: N=10000, M=1024
real               :: Q(N,N)
contains
function P(k,i)
!$omp declare simd uniform(i) linear(k) notinbranch
!$omp declare target
real               :: P
integer,intent(in) :: k,i
   P=(Q(k,i) * Q(i,k))
end function
end module
function accum() result(tmp)
use my_global_array
real    :: tmp, tmp1
integer :: i
   tmp = 0.0e0
   !$omp target
   !$omp parallel do private(tmp1) reduction(+:tmp)
   do i=1,N
      tmp1 = 0.0e0
      !$omp simd reduction(+:tmp1)
      do k = 1,M
         tmp1 = tmp1 + P(k,i)
      end do
      tmp = tmp + tmp1
   end do
   !$omp end target
end function
