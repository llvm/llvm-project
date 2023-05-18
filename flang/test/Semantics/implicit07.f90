! RUN: %python %S/test_errors.py %s %flang_fc1
implicit none(external)
external x
integer :: f, i, arr(1) = [0]
call x
!ERROR: 'y' is an external procedure without the EXTERNAL attribute in a scope with IMPLICIT NONE(EXTERNAL)
call y
!ERROR: 'f' is an external procedure without the EXTERNAL attribute in a scope with IMPLICIT NONE(EXTERNAL)
i = f()
block
  !ERROR: 'z' is an external procedure without the EXTERNAL attribute in a scope with IMPLICIT NONE(EXTERNAL)
  call z
end block
print *, arr(1) ! no error
end
