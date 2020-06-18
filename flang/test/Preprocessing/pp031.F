*  // C++ comment NOT erased from #define
      integer, parameter :: KWM = 666
#define KWM 777 // C comment
      if (KWM .eq. 777) then
        print *, 'pp031.F FAIL (should not have compiled)'
      else
        print *, 'pp031.F FAIL: ', KWM
      end if
      end
