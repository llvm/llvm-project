! @@name:	target.5f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
module params
integer,parameter :: THRESHOLD1=1000000, THRESHHOLD2=1000
end module
subroutine vec_mult(p, v1, v2, N)
   use params
   real    ::  p(N), v1(N), v2(N)
   integer ::  i
   call init(v1, v2, N)
   !$omp target if(N>THRESHHOLD1) map(to: v1, v2 ) map(from: p)
      !$omp parallel do if(N>THRESHOLD2)
      do i=1,N
	 p(i) = v1(i) * v2(i)
      end do
   !$omp end target
   call output(p, N)
end subroutine
