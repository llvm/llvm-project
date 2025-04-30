! @@name:	private.1f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	success
      PROGRAM PRIV_EXAMPLE
        INTEGER I, J

        I = 1
        J = 2

!$OMP   PARALLEL PRIVATE(I) FIRSTPRIVATE(J)
          I = 3
          J = J + 2
!$OMP   END PARALLEL

        PRINT *, I, J  ! I .eq. 1 .and. J .eq. 2
      END PROGRAM PRIV_EXAMPLE
