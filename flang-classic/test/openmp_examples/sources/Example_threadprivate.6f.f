! @@name:	threadprivate.6f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	success
      MODULE INC_MODULE_GOOD3
        REAL, POINTER :: WORK(:)
        SAVE WORK
!$OMP   THREADPRIVATE(WORK)
      END MODULE INC_MODULE_GOOD3

      SUBROUTINE SUB1(N)
      USE INC_MODULE_GOOD3
!$OMP   PARALLEL PRIVATE(THE_SUM)
        ALLOCATE(WORK(N))
        CALL SUB2(THE_SUM)
       WRITE(*,*)THE_SUM
!$OMP   END PARALLEL
      END SUBROUTINE SUB1

      SUBROUTINE SUB2(THE_SUM)
        USE INC_MODULE_GOOD3
        WORK(:) = 10
        THE_SUM=SUM(WORK)
      END SUBROUTINE SUB2

      PROGRAM INC_GOOD3
        N = 10
        CALL SUB1(N)
      END PROGRAM INC_GOOD3
