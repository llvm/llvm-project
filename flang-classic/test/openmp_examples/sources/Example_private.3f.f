! @@name:	private.3f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
      SUBROUTINE PRIV_EXAMPLE3()
        INTEGER I, A

!$OMP   PARALLEL PRIVATE(A)
         A = 1
!$OMP     PARALLEL DO PRIVATE(A)
          DO I = 1, 10
            A = 2
          END DO
!$OMP     END PARALLEL DO
        PRINT *, A ! Outer A still has value 1
!$OMP   END PARALLEL
      END SUBROUTINE PRIV_EXAMPLE3
