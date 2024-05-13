! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors for DREAL, DIMAG, DCONJG intrinsics

subroutine s()
  real :: a
  complex(4) :: c4 ! test scalar
  complex(8) :: c8
  complex(16) :: c16(2) ! test array

  !ERROR: Actual argument for 'a=' has bad type 'REAL(4)'
  print *, dreal(a)

  print *, dreal(c4)

  print *, dreal(c8)

  print *, dreal(c16)

  !ERROR: Actual argument for 'z=' has bad type 'REAL(4)'
  print *, dimag(a)

  !ERROR: Actual argument for 'z=' has bad type or kind 'COMPLEX(4)'
  print *, dimag(c4)

  print *, dimag(c8)

  !ERROR: Actual argument for 'z=' has bad type or kind 'COMPLEX(16)'
  print *, dimag(c16)

  !ERROR: Actual argument for 'z=' has bad type 'REAL(4)'
  print *, dconjg(a)

  !ERROR: Actual argument for 'z=' has bad type or kind 'COMPLEX(4)'
  print *, dconjg(c4)

  print *, dconjg(c8)

  !ERROR: Actual argument for 'z=' has bad type or kind 'COMPLEX(16)'
  print *, dconjg(c16)

end subroutine
