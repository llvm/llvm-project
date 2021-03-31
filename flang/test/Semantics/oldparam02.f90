! RUN: not %flang -falternative-parameter-statement -fsyntax-only %s 2>&1 | FileCheck %s

! Error tests for "old style" PARAMETER statements
subroutine subr(x1,x2,x3,x4,x5)
  type(*), intent(in) :: x1
  class(*), intent(in) :: x2
  real, intent(in) :: x3(*)
  real, intent(in) :: x4(:)
  character(*), intent(in) :: x5
  !CHECK: error: TYPE(*) dummy argument may only be used as an actual argument
  parameter p1 = x1
  !CHECK: error: Must be a constant value
  parameter p2 = x2
  !CHECK: error: Whole assumed-size array 'x3' may not appear here without subscripts
  parameter p3 = x3
  !CHECK: error: Must be a constant value
  parameter p4 = x4
  !CHECK: error: Must be a constant value
  parameter p5 = x5
  !CHECK: The expression must be a constant of known type
  parameter p6 = z'feedfacedeadbeef'
  !CHECK: error: Must be a constant value
  parameter p7 = len(x5)
  real :: p8
  !CHECK: error: Alternative style PARAMETER 'p8' must not already have an explicit type
  parameter p8 = 666
end
