!RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
subroutine s(a,n)
  real a(n)
!CHECK: INTEGER(KIND=8_4) n
  integer(int_ptr_kind()) n
end
