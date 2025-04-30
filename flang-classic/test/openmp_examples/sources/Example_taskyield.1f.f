! @@name:	taskyield.1f
! @@type:	F-free
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
subroutine foo ( lock, n )
   use omp_lib
   integer (kind=omp_lock_kind) :: lock
   integer n
   integer i

   do i = 1, n
     !$omp task
       call something_useful()
       do while ( .not. omp_test_lock(lock) )
         !$omp taskyield
       end do
       call something_critical()
       call omp_unset_lock(lock)
     !$omp end task
   end do

end subroutine
