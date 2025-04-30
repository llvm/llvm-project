! @@name:	target.4bf
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
module mults
contains
subroutine vec_mult(p,v1,v2,N)
   real,dimension(*) :: p, v1, v2
   integer           :: N,i
   call init(v1, v2, N)
   !$omp target map(to: v1(1:N), v2(:N)) map(from: p(1:N))
   !$omp parallel do
   do i=1,N
      p(i) = v1(i) * v2(i)
   end do
   call output(p, N)
   !$omp end target
end subroutine
end module
