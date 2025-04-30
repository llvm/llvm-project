! @@name:	lastprivate.1f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
      SUBROUTINE LASTPRIV(N, A, B)

        INTEGER N
        REAL A(*), B(*)
        INTEGER I
!$OMP PARALLEL
!$OMP DO LASTPRIVATE(I)

        DO I=1,N-1
          A(I) = B(I) + B(I+1)
        ENDDO

!$OMP END PARALLEL
        A(I) = B(I)      ! I has the value of N here

      END SUBROUTINE LASTPRIV
