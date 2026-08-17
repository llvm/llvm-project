! RUN: not %flang_fc1 -fopenmp -pedantic -Werror -fdebug-unparse %s 2>&1 | FileCheck %s --check-prefix=ERROR
! Continuation before C style comment.

integer :: i
i&
! ERROR: portability: nonstandard usage: C-style comment
/* c */ = &
! ERROR: portability: nonstandard usage: C-style comment
/* d */ 7
if (i /= 7) then
  print *, 'error', i
else
  print *, 'pass'
end if
end
