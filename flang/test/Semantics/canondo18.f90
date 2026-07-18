!RUN: %python %S/test_errors.py %s %flang_fc1
do 10 j=1,1
!ERROR: This statement cannot terminate the DO loop
10 stop
end
