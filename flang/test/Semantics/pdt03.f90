! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
type t(kp1,kp2)
  integer, kind :: kp1
  integer(kp1), kind :: kp2 = kp1
end type
type(t(kp1=8_8)) x
!CHECK: 4_4, 8_4, 8_4, 8_8
print *, kind(x%kp1), x%kp1, kind(x%kp2), x%kp2
end
