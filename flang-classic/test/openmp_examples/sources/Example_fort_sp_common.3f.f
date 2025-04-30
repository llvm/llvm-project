! @@name:	fort_sp_common.3f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
      SUBROUTINE COMMON_GOOD3()
        COMMON /C/ X,Y
!$OMP   PARALLEL PRIVATE (/C/)
          ! do work here
!$OMP   END PARALLEL
!$OMP   PARALLEL SHARED (/C/)
          ! do work here
!$OMP   END PARALLEL
      END SUBROUTINE COMMON_GOOD3
