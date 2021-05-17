! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: 100format(3hKWM)
! CHECK: if(ch.eq.'KWM')then
* KWM NOT expanded in Hollerith in FORMAT
#define KWM 666
#define HKWM 667
      character(len=3) :: ch
 100  format(3HKWM)
      write(ch, 100)
      if (ch .eq. 'KWM') then
        print *, 'pp024.F yes'
      else
        print *, 'pp024.F no: ', ch
      end if
      end
