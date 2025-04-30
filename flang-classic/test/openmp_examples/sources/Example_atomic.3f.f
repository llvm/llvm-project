! @@name:	atomic.3f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
       function fetch_and_add(p)
       integer:: fetch_and_add
       integer, intent(inout) :: p

! Atomically read the value of p and then increment it. The previous value is
! returned. This can be used to implement a simple lock as shown below.
!$omp atomic capture
       fetch_and_add = p
       p = p + 1
!$omp end atomic
       end function fetch_and_add
       module m
       interface
         function fetch_and_add(p)
           integer :: fetch_and_add
           integer, intent(inout) :: p
         end function
         function atomic_read(p)
           integer :: atomic_read
           integer, intent(in) :: p
         end function
       end interface
       type locktype
          integer ticketnumber
          integer turn
       end type
       contains
       subroutine do_locked_work(lock)
       type(locktype), intent(inout) :: lock
       integer myturn
       integer junk
! obtain the lock
        myturn = fetch_and_add(lock%ticketnumber)
        do while (atomic_read(lock%turn) .ne. myturn)
          continue
        enddo
! Do some work. The flush is needed to ensure visibility of variables
! not involved in atomic directives
!$omp flush
       call work
!$omp flush
! Release the lock
       junk = fetch_and_add(lock%turn)
       end subroutine
       end module
