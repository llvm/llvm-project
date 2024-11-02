! RUN: %python %S/test_errors.py %s %flang_fc1
! Shape conformance checks on assignments
program test
  real :: a0, a1a(2), a1b(3), a2a(2,3), a2b(3,2)
  a0 = 0. ! ok
  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches scalar REAL(4) and rank 1 array of REAL(4)
  a0 = [0.]
  a1a = 0. ! ok
  a1a = [(real(j),j=1,2)] ! ok
  !ERROR: Dimension 1 of left-hand side has extent 2, but right-hand side has extent 3
  a1a = [(real(j),j=1,3)]
  !ERROR: Dimension 1 of left-hand side has extent 3, but right-hand side has extent 2
  a1b = a1a
  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches rank 1 array of REAL(4) and rank 2 array of REAL(4)
  a1a = a2a
  a1a = a2a(:,1) ! ok
  a2a = 0. ! ok
  a2a(:,1) = a1a ! ok
  !ERROR: Dimension 1 of left-hand side has extent 3, but right-hand side has extent 2
  a2a(1,:) = a1a
  !ERROR: Dimension 1 of left-hand side has extent 2, but right-hand side has extent 3
  a2a = a2b
end
