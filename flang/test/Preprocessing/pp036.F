* #define FALSE TRUE ...  .FALSE. -> .TRUE.
#define FALSE TRUE
      if (.FALSE.) then
        print *, 'pp036.F pass'
      else
        print *, 'pp036.F FAIL: ', .FALSE.
      end if
      end
