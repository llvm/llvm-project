! RUN: %python %S/test_errors.py %s %flang_fc1

!ERROR: Label '600' was not found
if ( A ) 100, 200, 600
100 CONTINUE
200 CONTINUE
300 CONTINUE

!ERROR: Label '601' was not found
if ( A ) 101, 601, 301
101 CONTINUE
201 CONTINUE
301 CONTINUE

!ERROR: Label '602' was not found
if ( A ) 602, 202, 302
102 CONTINUE
202 CONTINUE
302 CONTINUE

END
