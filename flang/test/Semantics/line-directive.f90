!RUN: not %flang -fsyntax-only %s 2>&1 | FileCheck %s
subroutine s
  implicit none
!CHECK: line-directive.f90:5:3: error: No explicit type declared for 'a'
  a = 1.
#line 100
!CHECK: line-directive.f90:101:3: error: No explicit type declared for 'b'
  b = 2.
#line "sourceFile.cobol" 200
!CHECK: sourceFile.cobol:201:3: error: No explicit type declared for 'c'
  c = 3.
# 300
!CHECK: sourceFile.cobol:301:3: error: No explicit type declared for 'd'
  d = 4.
# "/dev/random" 400
!CHECK: random:401:3: error: No explicit type declared for 'e'
  e = 5.
end
