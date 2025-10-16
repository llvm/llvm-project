!RUN: %python %S/test_errors.py %s %flang_fc1
!ERROR: 'kind=' argument must be a constant scalar integer whose value is a supported kind for the intrinsic result type
print *, real(1.,666)
end
