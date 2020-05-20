! RUN: %S/test_errors.sh %s %t %f18
! Check calls with alt returns

       CALL TEST (N, *100, *200 )
       PRINT *,'Normal return'
       STOP
100    PRINT *,'First alternate return'
       STOP
200    PRINT *,'Secondnd alternate return'
       END
