! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: ch = 'KWM'
! CHECK: if (ch .eq. 'KWM') then
* KWM NOT expanded in 'literal'
#define KWM 666
      character(len=3) :: ch
      ch = 'KWM'
      if (ch .eq. 'KWM') then
        print *, 'pp021.F yes'
      else
        print *, 'pp021.F no: ', ch
      end if
      end
