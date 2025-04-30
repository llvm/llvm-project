! @@name:	workshare.2f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	no
! @@expect:	success
      SUBROUTINE WSHARE2(AA, BB, CC, DD, EE, FF, N)
      INTEGER N
      REAL AA(N,N), BB(N,N), CC(N,N)
      REAL DD(N,N), EE(N,N), FF(N,N)

!$OMP   PARALLEL
!$OMP     WORKSHARE
            AA = BB
            CC = DD
!$OMP     END WORKSHARE NOWAIT
!$OMP     WORKSHARE
            EE = FF
!$OMP     END WORKSHARE
!$OMP   END PARALLEL
       END SUBROUTINE WSHARE2
