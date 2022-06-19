!RUN: %flang -fsyntax-only %s 2>&1 | FileCheck %s
integer, parameter :: j = 10
! CHECK: Implied DO index 'j' uses an object of the same name in its bounds expressions
real :: a(10) = [(j, j=1,j)]
end
