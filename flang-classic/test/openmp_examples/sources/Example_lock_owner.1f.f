! @@name:	lock_owner.1f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	success
        program lock
        use omp_lib
        integer :: x
        integer (kind=omp_lock_kind) :: lck

        call omp_init_lock (lck)
        call omp_set_lock(lck)
        x = 0

!$omp parallel shared (x)
!$omp master
        x = x + 1
        call omp_unset_lock(lck)
!$omp end master

!       Some more stuff.
!$omp end parallel

        call omp_destroy_lock(lck)
        end
