! @@name:	threadprivate.2f
! @@type:	F-fixed
! @@compilable:	no
! @@linkable:	no
! @@expect:	failure
      MODULE INC_MODULE
        COMMON /T/ A
      END MODULE INC_MODULE

      SUBROUTINE INC_MODULE_WRONG()
        USE INC_MODULE
!$OMP   THREADPRIVATE(/T/)
      !non-conforming because /T/ not declared in INC_MODULE_WRONG
      END SUBROUTINE INC_MODULE_WRONG
! { error "PGF90-S-0155-THREADPRIVATE common block is empty - t" }
