! @@name:	device.2f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
subroutine vec_mult(p, v1, v2, N)
use omp_lib, ONLY : omp_get_num_devices
real    :: p(N), v1(N), v2(N)
integer :: N, i, ndev
logical :: do_offload
   call init(v1, v2, N)
   ndev = omp_get_num_devices()
   do_offload = (ndev>0) .and. (N>1000000)
   !$omp target if(do_offload) map(to: v1, v2) map(from: p)
   !$omp parallel do if(N>1000)
      do i=1,N
         p(i) = v1(i) * v2(i)
      end do
   !$omp end target
   call output(p, N)
end subroutine
