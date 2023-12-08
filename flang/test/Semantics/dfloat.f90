! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! Checks that a call to the legacy extension intrinsic function
! DFLOAT is transmogrified into a type conversion operation.
module m
  !CHECK: d = 1._8
  double precision :: d = dfloat(1)
 contains
  subroutine sub(n)
    integer, intent(in) :: n
    !CHECK: 2._8
    print *, dfloat(2)
    !CHECK: real(n,kind=8)
    print *, dfloat(n)
  end subroutine
end module
