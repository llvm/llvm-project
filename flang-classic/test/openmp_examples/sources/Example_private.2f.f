! @@name:	private.2f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
      MODULE PRIV_EXAMPLE2
        REAL A

        CONTAINS

          SUBROUTINE G(K)
            REAL K
            A = K  ! Accessed in the region but outside of the
                   ! construct; therefore unspecified whether
                   ! original or private list item is modified.
          END SUBROUTINE G

          SUBROUTINE F(N)
          INTEGER N
          REAL A

            INTEGER I
!$OMP       PARALLEL DO PRIVATE(A)
              DO I = 1,N
                A = I
                CALL G(A*2)
              ENDDO
!$OMP       END PARALLEL DO
          END SUBROUTINE F

      END MODULE PRIV_EXAMPLE2
