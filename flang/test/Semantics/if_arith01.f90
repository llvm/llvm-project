! RUN: %S/test_errors.sh %s %t %f18
! Check that a basic arithmetic if compiles.

if ( A ) 100, 200, 300
100 CONTINUE
200 CONTINUE
300 CONTINUE
END
