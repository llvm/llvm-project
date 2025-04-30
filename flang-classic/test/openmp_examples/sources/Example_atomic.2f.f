! @@name:	atomic.2f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
       function atomic_read(p)
       integer :: atomic_read
       integer, intent(in) :: p
! Guarantee that the entire value of p is read atomically. No part of
! p can change during the read operation.

!$omp atomic read
       atomic_read = p
       return
       end function atomic_read

       subroutine atomic_write(p, value)
       integer, intent(out) :: p
       integer, intent(in) :: value
! Guarantee that value is stored atomically into p. No part of p can change
! until after the entire write operation is completed.
!$omp atomic write
       p = value
       end subroutine atomic_write
