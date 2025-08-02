! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
function s(x) result(i)
!ERROR: internal error when processing function return
integer::x
procedure():: i
end function
end
