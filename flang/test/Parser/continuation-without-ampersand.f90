! RUN: %flang_fc1 -fsyntax-only -pedantic %s 2>&1 | FileCheck %s
! Continuation between repeated quotation marks
subroutine test
!CHECK: portability: Character literal continuation line should have been preceded by '&'
  print *, 'needs an '&
'ampersand'''
!CHECK: portability: Character literal continuation line should have been preceded by '&'
  print *, 'also needs an '&
 'ampersand'''
!CHECK-NOT: portability: Character literal continuation line should have been preceded by '&'
  print *, 'has an '&
&'ampersand'''
end
