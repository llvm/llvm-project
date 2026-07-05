! RUN: %python %S/test_errors.py %s %flang_fc1
!ERROR: No explicit type declared for 'func'
!ERROR: No explicit type declared for 'obj'
subroutine implicit_none(func, sub, obj)
  implicit none
  call sub ! ok
  print *, func()
  print *, obj
end
