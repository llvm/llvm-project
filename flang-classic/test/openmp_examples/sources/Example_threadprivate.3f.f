! @@name:	threadprivate.3f
! @@type:	F-fixed
! @@compilable:	no
! @@linkable:	no
! @@expect:	failure
      SUBROUTINE INC_WRONG()
        COMMON /T/ A
!$OMP   THREADPRIVATE(/T/)

        CONTAINS
          SUBROUTINE INC_WRONG_SUB()
!$OMP       PARALLEL COPYIN(/T/)
! { error "PGF90-S-0155-t is not a THREADPRIVATE common block" 12 }
! { error "PGF90-S-0038-Symbol, t, has not been explicitly declared" 12 }
      !non-conforming because /T/ not declared in INC_WRONG_SUB
!$OMP       END PARALLEL
          END SUBROUTINE INC_WRONG_SUB
      END SUBROUTINE INC_WRONG
