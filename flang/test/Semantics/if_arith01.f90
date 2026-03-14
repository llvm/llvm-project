! RUN: %python %S/test_errors.py %s %flang_fc1
! Check that a basic arithmetic if compiles.

if ( A ) 100, 200, 300
100 CONTINUE
200 CONTINUE
300 CONTINUE
END
