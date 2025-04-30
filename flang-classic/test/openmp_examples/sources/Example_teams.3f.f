! @@name:	teams.3f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
function dotprod(B,C,N) result(sum)
   real    :: B(N), C(N), sum
   integer :: N, i
   sum = 0.0e0
   !$omp target teams map(to: B, C)
   !$omp distribute parallel do reduction(+:sum)
      do i = 1,N
         sum = sum + B(i) * C(i)
      end do
   !$omp end target teams 
end function
