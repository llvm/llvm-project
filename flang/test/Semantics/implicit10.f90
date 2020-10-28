! RUN: %S/test_errors.sh %s %t %f18 -fimplicit-none-type-always

!ERROR: No explicit type declared for 'f'
function f()
  !ERROR: No explicit type declared for 'x'
  f = x
end
