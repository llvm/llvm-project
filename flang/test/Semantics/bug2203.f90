!RUN: %python %S/test_errors.py %s %flang_fc1
integer values(7)
!ERROR: VALUES= argument to DATE_AND_TIME must have at least 8 elements
call date_and_time(values=values)
end
