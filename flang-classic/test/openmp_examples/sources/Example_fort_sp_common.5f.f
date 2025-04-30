! @@name:	fort_sp_common.5f
! @@type:	F-fixed
! @@compilable:	no
! @@linkable:	no
! @@expect:	failure
      SUBROUTINE COMMON_WRONG2()
        COMMON /C/ X,Y
! Incorrect: common block C cannot be declared both
! shared and private
!$OMP   PARALLEL PRIVATE (/C/), SHARED(/C/)
! { error "PGF90-S-0155-x is used in multiple data sharing clauses" 10 }
! { error "PGF90-S-0155-y is used in multiple data sharing clauses" 10 }
          ! do work here
!$OMP   END PARALLEL

      END SUBROUTINE COMMON_WRONG2
