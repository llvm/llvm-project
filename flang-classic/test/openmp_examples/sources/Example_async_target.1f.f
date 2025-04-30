! @@name:	async_target.1f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
module parameters
integer, parameter :: N=1000000000, CHUNKSZ=1000000
end module
subroutine pipedF()
use parameters, ONLY: N, CHUNKSZ
integer            :: C, i
real               :: z(N)

interface
   function F(z)
   !$omp declare target
     real, intent(IN) ::z
     real             ::F
   end function F
end interface

   call init(z,N)

   do C=1,N,CHUNKSZ

      !$omp task
      !$omp target map(z(C:C+CHUNKSZ-1))
      !$omp parallel do
         do i=C,C+CHUNKSZ-1
            z(i) = F(z(i))
         end do
      !$omp end target
      !$omp end task

   end do
   !$omp taskwait
   print*, z

end subroutine pipedF
