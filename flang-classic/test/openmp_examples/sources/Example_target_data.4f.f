! @@name:	target_data.4f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
module mults
contains
subroutine foo(p0,v1,v2,N)
real,pointer,dimension(:) :: p0, v1, v2
integer                   :: N,i
   call init(v1, v2, N)
   !$omp target data map(to: v1, v2) map(from: p0)
    call vec_mult(p0,v1,v2,N)
   !$omp end target data
   call output(p0, N)
end subroutine
subroutine vec_mult(p1,v3,v4,N)
real,pointer,dimension(:) :: p1, v3, v4
integer                   :: N,i
   !$omp target map(to: v3, v4) map(from: p1)
   !$omp parallel do
   do i=1,N
      p1(i) = v3(i) * v4(i)
   end do
   !$omp end target
end subroutine
end module
