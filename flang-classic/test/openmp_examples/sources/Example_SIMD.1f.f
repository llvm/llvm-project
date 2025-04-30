! @@name:	SIMD.1f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
subroutine star(a,b,c,n,ioff_ptr)
   implicit none
   double precision :: a(*),b(*),c(*)
   integer          :: n, i
   integer, pointer :: ioff_ptr
 
   !$omp simd
   do i = 1,n
      a(i) = a(i) * b(i) * c(i+ioff_ptr)
   end do
 
end subroutine
