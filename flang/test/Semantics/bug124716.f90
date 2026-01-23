! RUN: %python %S/test_modfile.py %s %flang_fc1
MODULE m1
  INTERFACE
    MODULE SUBROUTINE sub1(N, ARR)
      INTEGER, INTENT(IN) :: N
      INTEGER, DIMENSION(N) :: ARR
    END SUBROUTINE
  END INTERFACE
END MODULE
SUBMODULE (m1) m1sub
 CONTAINS
  MODULE SUBROUTINE sub1(N, ARR)
    INTEGER, INTENT(IN) :: N
    INTEGER, DIMENSION(N) :: ARR
    PRINT *, "sub1", N, ARR
  END SUBROUTINE
END SUBMODULE

!Expect: m1.mod
!module m1
!interface
!module subroutine sub1(n,arr)
!integer(4),intent(in)::n
!integer(4)::arr(1_8:int(n,kind=8))
!end
!end interface
!end

!Expect: m1-m1sub.mod
!submodule(m1) m1sub
!contains
!module subroutine sub1(n,arr)
!integer(4),intent(in)::n
!integer(4)::arr(1_8:int(n,kind=8))
!end
!end
