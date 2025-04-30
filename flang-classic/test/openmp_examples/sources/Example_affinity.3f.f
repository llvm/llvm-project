! @@name:	affinity.3f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	success
      PROGRAM EXAMPLE
!$OMP PARALLEL PROC_BIND(CLOSE) NUM_THREADS(4)
      CALL WORK()
!$OMP END PARALLEL
      END PROGRAM EXAMPLE
