! @@name:	fort_sp_common.2f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
      SUBROUTINE COMMON_GOOD2()
        COMMON /C/ X,Y
        REAL X, Y
        INTEGER I
!$OMP   PARALLEL
!$OMP     DO PRIVATE(/C/)
          DO I=1,1000
            ! do work here
          ENDDO
!$OMP     END DO
!$OMP     DO PRIVATE(X)
          DO I=1,1000
            ! do work here
          ENDDO
!$OMP     END DO
!$OMP   END PARALLEL
      END SUBROUTINE COMMON_GOOD2
