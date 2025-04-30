! @@name:	SIMD.3f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
subroutine work( a, b, n, sum )
   implicit none
   integer :: i, n
   double precision :: a(n), b(n), sum, tmp

   sum = 0.0d0
   !$omp simd private(tmp) reduction(+:sum)
   do i = 1,n
      tmp = a(i) + b(i)
      sum = sum + tmp
   end do

end subroutine work
