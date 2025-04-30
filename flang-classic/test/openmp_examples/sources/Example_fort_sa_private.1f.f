! @@name:	fort_sa_private.1f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	rt-error
       SUBROUTINE SUB()
       COMMON /BLOCK/ X
       PRINT *,X             ! X is undefined
       END SUBROUTINE SUB

       PROGRAM PRIV_RESTRICT
         COMMON /BLOCK/ X
         X = 1.0
!$OMP    PARALLEL PRIVATE (X)
         X = 2.0
         CALL SUB()
!$OMP    END PARALLEL
      END PROGRAM PRIV_RESTRICT
