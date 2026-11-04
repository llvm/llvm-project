! RUN: %python %S/test_errors.py %s %flang_fc1
external ext
integer caf[*]
!ERROR: OPERATION= argument of CO_REDUCE() must be a pure function of two data arguments
call co_reduce(caf, ext)
end
