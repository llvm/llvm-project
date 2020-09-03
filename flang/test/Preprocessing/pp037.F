* fixed-form clipping NOT applied to #define
      integer, parameter :: KWM = 666
*        1         2         3         4         5         6         7
*234567890123456789012345678901234567890123456789012345678901234567890123
#define KWM                                                          7777
      if (KWM .eq. 777) then
        print *, 'pp037.F pass'
      else
        print *, 'pp037.F FAIL: ', KWM
      end if
      end
