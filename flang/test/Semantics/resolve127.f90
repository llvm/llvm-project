!RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
double precision, parameter :: x = 1.0d0
type t
  !CHECK: REAL :: x(1_4) = [INTEGER(4)::8_4]
  real :: x(1) = [(kind(x),j=1,1)]
end type
end
