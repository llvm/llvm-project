! @@name:	declare_target.4f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
module my_global_array
!$omp declare target (N,Q)
integer, parameter :: N=10
real               :: Q(N,N)
contains
function Pfun(i,k)
!$omp declare target
real               :: Pfun
integer,intent(in) :: i,k
   Pfun=(Q(i,k) * Q(k,i))
end function
end module
function accum(k) result(tmp)
use my_global_array
real    :: tmp
integer :: i, k
   tmp = 0.0e0
   !$omp target
   !$omp parallel do reduction(+:tmp)
   do i=1,N
      tmp = tmp + Pfun(k,i)
   end do
   !$omp end target
end function
