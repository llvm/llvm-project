! @@name:	target_data.5f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
module my_mult
contains
subroutine foo(p0,v1,v2,N)
real,dimension(:) :: p0, v1, v2
integer           :: N,i
   call init(v1, v2, N)
   !$omp target data map(to: v1, v2) map(from: p0)
    call vec_mult(p0,v1,v2,N)
   !$omp end target data
   call output(p0, N)
end subroutine
subroutine vec_mult(p1,v3,v4,N)
real,dimension(:) :: p1, v3, v4
integer           :: N,i
   !$omp target map(to: v3, v4) map(from: p1)
   !$omp parallel do
   do i=1,N
      p1(i) = v3(i) * v4(i)
   end do
   !$omp end target
end subroutine
end module
program main
use my_mult
integer, parameter :: N=1024
real,allocatable, dimension(:) :: p, v1, v2
   allocate( p(N), v1(N), v2(N) )
   call foo(p,v1,v2,N)
end program
