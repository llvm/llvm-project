! RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s

! Test ignoring @PROCESS directive in fixed source form

@process opt(3)
@process	opt(0)
@process
@processopt(3)
      subroutine f()
c@process
      end

!CHECK: Character in fixed-form label field must be a digit
@

!CHECK: Character in fixed-form label field must be a digit
@proce 

!CHECK: Character in fixed-form label field must be a digit
@precoss 
