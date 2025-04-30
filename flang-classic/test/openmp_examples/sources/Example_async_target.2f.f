! @@name:	async_target.2f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
 subroutine mult(p,  N, idev)
   use omp_lib, ONLY: omp_is_initial_device
   real             :: p(N)
   real,allocatable :: v1(:), v2(:)
   integer ::  i, idev
   !$omp declare target (init)

   !$omp task shared(v1,v2) depend(out: N)
      !$omp target device(idev) map(v1,v2)
         if( omp_is_initial_device() ) &
            stop "not executing on target device"
         allocate(v1(N), v2(N))
         call init(v1,v2,N)
      !$omp end target
   !$omp end task

   call foo()  ! execute other work asychronously

   !$omp task shared(v1,v2,p) depend(in: N)
      !$omp target device(idev) map(to: v1,v2) map(from: p)
         if( omp_is_initial_device() ) &
            stop "not executing on target device"
         !$omp parallel do
            do i = 1,N
               p(i) = v1(i) * v2(i)
            end do
         deallocate(v1,v2)

      !$omp end target
   !$omp end task

   !$omp taskwait
   call output(p, N)

end subroutine
