! @@name:	target_data.6f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
module params
integer,parameter :: THRESHOLD=1000000
end module
subroutine vec_mult(p, v1, v2, N)
   use params
   real    ::  p(N), v1(N), v2(N)
   integer ::  i
   call init(v1, v2, N)
   !$omp target data if(N>THRESHOLD) map(from: p)
      !$omp target if(N>THRESHOLD) map(to: v1, v2)
         !$omp parallel do
         do i=1,N
            p(i) = v1(i) * v2(i)
         end do
      !$omp end target
      call init_again(v1, v2, N)
      !$omp target if(N>THRESHOLD) map(to: v1, v2)
         !$omp parallel do
         do i=1,N
            p(i) = p(i) + v1(i) * v2(i)
         end do
      !$omp end target
   !$omp end target data
   call output(p, N)
end subroutine
