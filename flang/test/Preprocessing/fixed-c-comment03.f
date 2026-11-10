! RUN: %flang_fc1 -fsyntax-only %s
! RUN: not %flang_fc1 -cpp -fsyntax-only -pedantic -Werror %s 2>&1 | \
! RUN:    FileCheck %s --check-prefix=CPP-ERROR

      program p
      integer :: x, y
      y = 3
      x = y
! CPP-ERROR: fixed-c-comment03.f:11:6: portability: nonstandard usage: C-style comment
! CPP-ERROR: fixed-c-comment03.f:13:29: error: Incomplete character literal
     /* 2
      print *, x
      print *, 'tail */ text'

      x = 7
      write(*,10) x, x+1, x+2
! CPP-ERROR: fixed-c-comment03.f:20:13: error: Unmatched '('
! CPP-ERROR: fixed-c-comment03.f:20:19: portability: nonstandard usage: C-style comment
! CPP-ERROR: fixed-c-comment03.f:21:29: error: Incomplete character literal
   10 format(1x,i2/*(i3))
      print *, 'tail */ text'
      end
