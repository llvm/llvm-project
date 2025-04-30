! @@name:	SIMD.7f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	success
program fibonacci
   implicit none
   integer,parameter :: N=45
   integer           :: a(0:N-1), b(0:N-1)
   integer           :: i
   integer, external :: fib

   !$omp simd
   do i = 0,N-1
      b(i) = i
   end do

   !$omp simd
   do i=0,N-1
      a(i) = fib(b(i))
   end do
   
   write(*,*) "Done a(", N-1, ") = ", a(N-1)
                        ! 44  701408733
end program

recursive function fib(n) result(r)
!$omp declare simd(fib) inbranch
   implicit none
   integer  :: n, r

   if (n <= 1) then
      r = n
   else 
      r = fib(n-1) + fib(n-2)
   endif

end function fib
