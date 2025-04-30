! @@name:	target_update.2f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
subroutine vec_mult(p, v1, v2, N)
   interface
      logical function maybe_init_again (v1, N)
      real :: v1(N)
      integer :: N
      end function
   end interface
   real    ::  p(N), v1(N), v2(N)
   integer ::  i
   logical :: changed
   call init(v1, v2, N)
   !$omp target data map(to: v1, v2) map(from: p)
      !$omp target
         !$omp parallel do
         do i=1, N
            p(i) = v1(i) * v2(i)
         end do
      !$omp end target
      changed = maybe_init_again(v1, N)
      !$omp target update if(changed)  to(v1(:N))
      changed = maybe_init_again(v2, N)
      !$omp target update if(changed) to(v2(:N))
      !$omp target
         !$omp parallel do
         do i=1, N
            p(i) = p(i) + v1(i) * v2(i)
         end do
      !$omp end target
   !$omp end target data
   call output(p, N)
end subroutine
