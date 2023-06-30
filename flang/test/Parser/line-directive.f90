! RUN: %flang_fc1 -E %s 2>&1 | FileCheck %s
!CHECK: #line "{{.*[/\\]}}line-directive.f90" 3
subroutine s
  implicit none
  a = 1.
#line 100
!CHECK: #line 101
  b = 2.
#line "sourceFile.cobol" 200
!CHECK: #line "sourceFile.cobol" 201
  c = 3.
# 300
!CHECK: #line 301
  d = 4.
# "/dev/random" 400
!CHECK: #line "/dev/random" 401
  e = 5.
end
