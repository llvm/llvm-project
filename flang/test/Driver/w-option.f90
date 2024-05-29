! Test the default setting. Emit warnings only.
! RUN: %flang -c %s 2>&1 | FileCheck %s -check-prefix=DEFAULT

! Test that the warnings are not generated with `-w` option.
! RUN: %flang -c -w %s 2>&1 | FileCheck --allow-empty %s -check-prefix=WARNING

! Test that warnings are portability messages are generated.
! RUN: %flang -c -pedantic %s 2>&1 | FileCheck %s -check-prefixes=DEFAULT,PORTABILITY

! Test that warnings and portability messages are not generated.
! TODO: Support the last flag wins behaviour.
! RUN: %flang -c -pedantic -w %s 2>&1 | FileCheck --allow-empty %s -check-prefixes=WARNING,PORTABILITY-WARNING
! RUN: %flang -c -w -pedantic %s 2>&1 | FileCheck --allow-empty %s -check-prefixes=WARNING,PORTABILITY-WARNING
! DEFAULT: warning: Label '40' is in a construct that should not be used as a branch target here
! DEFAULT: warning: Label '50' is in a construct that should not be used as a branch target here
! WARNING-NOT: warning
! PORTABILITY: portability: Statement function 'sf1' should not contain an array constructor
! PORTABILITY-WARNING-NOT: portability

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
