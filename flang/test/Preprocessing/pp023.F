! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: ch=3hKWM
! CHECK: if(ch.eq.'KWM')then
* KWM NOT expanded in 9HHOLLERITH literal
#define KWM 666
#define HKWM 667
      character(len=3) :: ch
      ch = 3HKWM
      if (ch .eq. 'KWM') then
        print *, 'pp023.F yes'
      else
        print *, 'pp023.F no: ', ch
      end if
      end
