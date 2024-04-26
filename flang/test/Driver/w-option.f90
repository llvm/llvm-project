! RUN: %flang -c %s 2>&1 | FileCheck %s
! RUN: %flang -c -w %s 2>&1 | FileCheck --allow-empty %s -check-prefix=CHECK-W
! RUN: %flang -c -pedantic %s 2>&1 | FileCheck %s -check-prefixes=CHECK,CHECK-PORT
! RUN: %flang -c -pedantic -w %s 2>&1 | FileCheck --allow-empty %s -check-prefixes=CHECK-W,CHECK-PORT-W
! RUN: %flang -c -w -pedantic %s 2>&1 | FileCheck  --allow-empty %s -check-prefixes=CHECK-W,CHECK-PORT-W
! CHECK: warning: Label '40' is in a construct that should not be used as a branch target here
! CHECK: warning: Label '50' is in a construct that should not be used as a branch target here
! CHECK-W-NOT: warning
! CHECK-PORT: portability: Statement function 'sf1' should not contain an array constructor
! CHECK-PORT-W-NOT: portability

subroutine sub01(n)
  integer n
  GOTO (40,50,60) n
  if (n .eq. 1) then
40   print *, "xyz"
50 end if
60 continue
end subroutine sub01

subroutine sub02
  sf1(n) = sum([(j,j=1,n)])
end subroutine sub02
