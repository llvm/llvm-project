! @@name:	nesting_restrict.2f
! @@type:	F-fixed
! @@compilable:	maybe
! @@linkable:	no
! @@expect:	failure
       SUBROUTINE WORK1(I,N)
       INTEGER I, N
       INTEGER J
!$OMP   DO      ! incorrect nesting of loop regions
        DO J = 1, N
          CALL WORK(I,J)
        END DO
       END SUBROUTINE WORK1
       SUBROUTINE WRONG2(N)
       INTEGER N
       INTEGER I
!$OMP   PARALLEL DEFAULT(SHARED)
!$OMP     DO
          DO I = 1, N
            CALL WORK1(I,N)
          END DO
!$OMP   END PARALLEL
       END SUBROUTINE WRONG2
