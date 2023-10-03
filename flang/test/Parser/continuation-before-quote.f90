! RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! Continuation between repeated quotation marks
subroutine test
!CHECK: portability: Repeated quote mark in character literal continuation line should have been preceded by '&'
  print *, 'needs an '&
'ampersand'''
!CHECK-NOT: portability: Repeated quote mark in character literal continuation line should have been preceded by '&'
  print *, 'has an '&
&'ampersand'''
end
