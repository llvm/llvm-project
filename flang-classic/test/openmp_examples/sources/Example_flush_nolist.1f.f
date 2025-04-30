! @@name:	flush_nolist.1f
! @@type:	F-fixed
! @@compilable:	yes
! @@linkable:	yes
! @@expect:	success
      SUBROUTINE F1(Q)
        COMMON /DATA/ X, P
        INTEGER, TARGET  :: X
        INTEGER, POINTER :: P
        INTEGER Q

        Q = 1
!$OMP   FLUSH
        ! X, P and Q are flushed
        ! because they are shared and accessible
      END SUBROUTINE F1

      SUBROUTINE F2(Q)
        COMMON /DATA/ X, P
        INTEGER, TARGET  :: X
        INTEGER, POINTER :: P
        INTEGER Q

!$OMP   BARRIER
          Q = 2
!$OMP   BARRIER
          ! a barrier implies a flush
          ! X, P and Q are flushed
          ! because they are shared and accessible
      END SUBROUTINE F2

      INTEGER FUNCTION G(N)
        COMMON /DATA/ X, P
        INTEGER, TARGET  :: X
        INTEGER, POINTER :: P
        INTEGER N
        INTEGER I, J, SUM

        I = 1
        SUM = 0
        P = 1
!$OMP   PARALLEL REDUCTION(+: SUM) NUM_THREADS(10)
          CALL F1(J)
            ! I, N and SUM were not flushed
            !   because they were not accessible in F1
            ! J was flushed because it was accessible
          SUM = SUM + J

          CALL F2(J)
            ! I, N, and SUM were not flushed
            !   because they were not accessible in f2
            ! J was flushed because it was accessible
          SUM = SUM + I + J + P + N
!$OMP   END PARALLEL

        G = SUM
      END FUNCTION G

      PROGRAM FLUSH_NOLIST
        COMMON /DATA/ X, P
        INTEGER, TARGET  :: X
        INTEGER, POINTER :: P
        INTEGER RESULT, G

        P => X
        RESULT = G(7)
        PRINT *, RESULT
      END PROGRAM FLUSH_NOLIST
