! @@name:	SIMD.2f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	success
program main
   implicit none
   integer, parameter :: N=32
   integer :: i
   double precision   :: a(N), b(N)
   do i = 1,N
      a(i) = i-1
      b(i) = N-(i-1)
   end do
   call work(a, b, N )
   do i = 1,N
      print*, i,a(i)
   end do
end program

function add1(a,b,fact) result(c)
!$omp declare simd(add1) uniform(fact)
   implicit none
   double precision :: a,b,fact, c
   c = a + b + fact
end function

function add2(a,b,i, fact) result(c)
!$omp declare simd(add2) uniform(a,b,fact) linear(i:1)
   implicit none
   integer          :: i
   double precision :: a(*),b(*),fact, c
   c = a(i) + b(i) + fact
end function

subroutine work(a, b, n )
   implicit none
   double precision           :: a(n),b(n), tmp
   integer                    :: n, i
   double precision, external :: add1, add2

   !$omp simd private(tmp)
   do i = 1,n
      tmp  = add1(a(i), b(i), 1.0d0)
      a(i) = add2(a,    b, i, 1.0d0) + tmp
      a(i) = a(i) + b(i) + 1.0d0
   end do
end subroutine
