! RUN: %python %S/test_errors.py %s %flang_fc1
! Check calls with alt returns

       CALL TEST (N, *100, *200 )
       PRINT *,'Normal return'
       STOP
100    PRINT *,'First alternate return'
       STOP
200    PRINT *,'Secondnd alternate return'
       END
