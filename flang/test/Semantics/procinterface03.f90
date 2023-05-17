! RUN: %python %S/test_errors.py %s %flang_fc1
procedure(sin), pointer :: pp
!ERROR: 'pp' has an explicit interface and may not also have a type
real :: pp
end
