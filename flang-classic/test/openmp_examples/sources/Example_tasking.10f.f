! @@name:	tasking.10f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
       module example
       include 'omp_lib.h'
       integer (kind=omp_lock_kind) lock
       integer i
       contains
       subroutine work
       call omp_init_lock(lock)
!$omp parallel
     !$omp do
      do i=1,100
         !$omp task
              ! Outer task
              call omp_set_lock(lock)    ! lock is shared by
                                         ! default in the task
                     ! Capture data for the following task
                     !$omp task     ! Task Scheduling Point 1
                              ! do work here
                     !$omp end task
               call omp_unset_lock(lock)
         !$omp end task
      end do
!$omp end parallel
      call omp_destroy_lock(lock)
      end subroutine
      end module
