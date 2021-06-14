! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell

! Check errors from illegal (10.1.12 para 2) forward references
! in specification expressions to entities declared later in the
! same specification part.

module m1
  integer :: m1j1, m1j2, m1j3, m1j4
 contains
  subroutine s1
    !ERROR: Forward reference to 'm1j1' is not allowed in the same specification part
    integer(kind=kind(m1j1)) :: t_s1m1j1
    integer(kind=kind(m1s1j1)) :: t_s1j1 ! implicitly typed in s1
    integer :: m1j1, m1s1j1, m1s1j2, m1s1j4
    block
      !ERROR: Forward reference to 'm1j2' is not allowed in the same specification part
      integer(kind=kind(m1j2)) :: t_s1bm1j2
      !ERROR: Forward reference to 'm1s1j2' is not allowed in the same specification part
      integer(kind=kind(m1s1j2)) :: t_s1bm1s1j2
      !ERROR: Forward reference to 'm1s1j3' is not allowed in the same specification part
      integer(kind=kind(m1s1j3)) :: t_m1s1j3 ! m1s1j3 implicitly typed in s1
      integer :: m1j2, m1s1j2, m1s1j3
    end block
   contains
    subroutine s2
      !ERROR: Forward reference to 'm1j3' is not allowed in the same specification part
      integer(kind=kind(m1j3)) :: t_m1j3
      !ERROR: Forward reference to 'm1s1j3' is not allowed in the same specification part
      integer(kind=kind(m1s1j3)) :: t_m1s1j3
      integer :: m1j3, m1s1j3, m1s2j1
      block
        !ERROR: Forward reference to 'm1j4' is not allowed in the same specification part
        integer(kind=kind(m1j4)) :: t_m1j4
        !ERROR: Forward reference to 'm1s1j4' is not allowed in the same specification part
        integer(kind=kind(m1s1j4)) :: t_m1s1j4
        !ERROR: Forward reference to 'm1s2j1' is not allowed in the same specification part
        integer(kind=kind(m1s2j1)) :: t_m1s2j1
        !ERROR: Forward reference to 'm1s2j2' is not allowed in the same specification part
        integer(kind=kind(m1s2j2)) :: t_m1s2j2 ! m1s2j2 implicitly typed in s2
        integer :: m1j4, m1s1j4, m1s2j1, m1s2j2
      end block
    end subroutine
  end subroutine
end module

module m2
  implicit none
  integer :: m2j1, m2j2, m2j3, m2j4
 contains
  subroutine s1
    !ERROR: Forward reference to 'm2j1' is not allowed in the same specification part
    integer(kind=kind(m2j1)) :: t_s1m2j1
    !ERROR: No explicit type declared for 'm2s1j1'
    integer(kind=kind(m2s1j1)) :: t_s1j1
    integer :: m2j1, m2s1j1, m2s1j2, m2s1j4
    block
      !ERROR: Forward reference to 'm2j2' is not allowed in the same specification part
      integer(kind=kind(m2j2)) :: t_s1bm2j2
      !ERROR: Forward reference to 'm2s1j2' is not allowed in the same specification part
      integer(kind=kind(m2s1j2)) :: t_s1bm2s1j2
      !ERROR: No explicit type declared for 'm2s1j3'
      integer(kind=kind(m2s1j3)) :: t_m2s1j3
      integer :: m2j2, m2s1j2, m2s1j3
    end block
   contains
    subroutine s2
      !ERROR: Forward reference to 'm2j3' is not allowed in the same specification part
      integer(kind=kind(m2j3)) :: t_m2j3
      !ERROR: No explicit type declared for 'm2s1j3'
      integer(kind=kind(m2s1j3)) :: t_m2s1j3
      integer :: m2j3, m2s1j3, m2s2j1
      block
        !ERROR: Forward reference to 'm2j4' is not allowed in the same specification part
        integer(kind=kind(m2j4)) :: t_m2j4
        !ERROR: Forward reference to 'm2s1j4' is not allowed in the same specification part
        integer(kind=kind(m2s1j4)) :: t_m2s1j4
        !ERROR: Forward reference to 'm2s2j1' is not allowed in the same specification part
        integer(kind=kind(m2s2j1)) :: t_m2s2j1
        !ERROR: No explicit type declared for 'm2s2j2'
        integer(kind=kind(m2s2j2)) :: t_m2s2j2
        integer :: m2j4, m2s1j4, m2s2j1, m2s2j2
      end block
    end subroutine
  end subroutine
end module

! Case that elicited bad errors
SUBROUTINE KEEL
  INTEGER NODES
 CONTAINS
  SUBROUTINE SGEOM
    REAL :: RADIUS(nodes)
  END SUBROUTINE
END SUBROUTINE KEEL
