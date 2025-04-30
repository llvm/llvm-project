! @@name:	SIMD.5f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
subroutine work( a, b, c,  n )
   implicit none
   integer :: i,j,n
   double precision :: a(n,n), b(n,n), c(n,n), tmp

   !$omp do simd collapse(2) private(tmp)
   do j = 1,n
      do i = 1,n
         tmp = a(i,j) + b(i,j)
         c(i,j) = tmp 
      end do
   end do

end subroutine work
