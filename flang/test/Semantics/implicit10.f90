! RUN: %S/test_errors.sh %s %t %flang_fc1 -fimplicit-none
! REQUIRES: shell

!ERROR: No explicit type declared for 'f'
function f()
  !ERROR: No explicit type declared for 'x'
  f = x
end
