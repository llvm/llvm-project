!RUN: %python %S/test_errors.py %s %flang_fc1
procedure(), pointer :: pp => tan
!ERROR: EXTERNAL attribute was already specified on 'pp'
!ERROR: POINTER attribute was already specified on 'pp'
!ERROR: 'pp' was previously initialized
procedure(real), pointer :: pp => tan
end
