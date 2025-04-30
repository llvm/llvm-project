! @@name:	target.1f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
subroutine vec_mult(N)
   integer ::  i,N
   real    ::  p(N), v1(N), v2(N)
   call init(v1, v2, N)
   !$omp target
   !$omp parallel do
   do i=1,N
      p(i) = v1(i) * v2(i)
   end do
   !$omp end target
   call output(p, N)
end subroutine
