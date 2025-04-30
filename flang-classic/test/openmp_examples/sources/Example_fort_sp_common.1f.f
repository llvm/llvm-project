! @@name:	fort_sp_common.1f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
      SUBROUTINE COMMON_GOOD()
        COMMON /C/ X,Y
        REAL X, Y

!$OMP   PARALLEL PRIVATE (/C/)
          ! do work here
!$OMP   END PARALLEL
!$OMP   PARALLEL SHARED (X,Y)
          ! do work here
!$OMP   END PARALLEL
      END SUBROUTINE COMMON_GOOD
