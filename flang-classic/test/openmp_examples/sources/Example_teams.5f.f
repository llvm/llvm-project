! @@name:	teams.5f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
subroutine vec_mult(p, v1, v2, N)
   real    ::  p(N), v1(N), v2(N)
   integer ::  i
   call init(v1, v2, N)
   !$omp target teams map(to: v1, v2) map(from: p)
      !$omp distribute simd
         do i=1,N
            p(i) = v1(i) * v2(i)
         end do
   !$omp end target teams
   call output(p, N)
end subroutine
