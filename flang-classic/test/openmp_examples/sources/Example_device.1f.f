! @@name:	device.1f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	success
module params
   integer,parameter :: N=1024
end module params
module vmult
contains
   subroutine vec_mult(p, v1, v2, N)
   use omp_lib, ONLY : omp_is_initial_device
   !$omp declare target
   real    :: p(N), v1(N), v2(N)
   integer :: i, nthreads, N
      if (.not. omp_is_initial_device()) then
         print*, "1024 threads on target device"
         nthreads = 1024
      else
         print*, "8 threads on initial device"
         nthreads = 8
      endif
      !$omp parallel do private(i) num_threads(nthreads)
      do i = 1,N
        p(i) = v1(i) * v2(i)
      end do
   end subroutine vec_mult
end module vmult
program prog_vec_mult
use params
use vmult
real :: p(N), v1(N), v2(N)
   call init(v1,v2,N)
   !$omp target device(42) map(p, v1, v2)
      call vec_mult(p, v1, v2, N)
   !$omp end target
   call output(p, N)
end program
