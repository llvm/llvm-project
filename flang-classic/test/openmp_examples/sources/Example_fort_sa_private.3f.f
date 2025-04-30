! @@name:	fort_sa_private.3f
! @@type:	F-fixed
! @@compilable:	maybe
! @@linkable:	maybe
! @@expect:	rt-error
      PROGRAM PRIV_RESTRICT3
        EQUIVALENCE (X,Y)
        X = 1.0

!$OMP   PARALLEL PRIVATE(X)
          PRINT *,Y                  ! Y is undefined
          Y = 10
          PRINT *,X                  ! X is undefined
!$OMP   END PARALLEL
      END PROGRAM PRIV_RESTRICT3
