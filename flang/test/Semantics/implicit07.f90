! RUN: %S/test_errors.sh %s %t %f18
implicit none(external)
external x
call x
!ERROR: 'y' is an external procedure without the EXTERNAL attribute in a scope with IMPLICIT NONE(EXTERNAL)
call y
block
  !ERROR: 'z' is an external procedure without the EXTERNAL attribute in a scope with IMPLICIT NONE(EXTERNAL)
  call z
end block
end
