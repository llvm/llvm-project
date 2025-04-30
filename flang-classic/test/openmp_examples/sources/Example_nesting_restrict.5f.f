! @@name:	nesting_restrict.5f
! @@type:	F-fixed
! @@compilable:	no
! @@linkable:	no
! @@expect:	failure
      SUBROUTINE WRONG5(N)
      INTEGER N

!$OMP   PARALLEL DEFAULT(SHARED)
!$OMP     CRITICAL
            CALL WORK(N,1)
! incorrect nesting of barrier region in a critical region
!$OMP       BARRIER ! { error "PGF90-S-0155-Illegal context for barrier" }
            CALL WORK(N,2)
!$OMP     END CRITICAL
!$OMP   END PARALLEL
      END SUBROUTINE WRONG5
