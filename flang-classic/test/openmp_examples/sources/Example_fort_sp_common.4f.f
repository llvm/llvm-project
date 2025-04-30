! @@name:	fort_sp_common.4f
! @@type:	F-fixed
! @@compilable:	no
! @@linkable:	no
! @@expect:	failure
      SUBROUTINE COMMON_WRONG()
        COMMON /C/ X,Y
! Incorrect because X is a constituent element of C
!$OMP   PARALLEL PRIVATE(/C/), SHARED(X) ! { error "PGF90-S-0155-x is used in multiple data sharing clauses" }
          ! do work here
!$OMP   END PARALLEL
      END SUBROUTINE COMMON_WRONG
