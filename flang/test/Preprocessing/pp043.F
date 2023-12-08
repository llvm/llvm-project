! RUN: %flang -E %s 2>&1 | FileCheck %s
! CHECK: if (KWM .eq. 777) then
* #define with # in column 6 is a continuation line in fixed-form
      integer, parameter :: defineKWM666 = 555
      integer, parameter :: KWM =
     #define KWM 666
     ++222
      if (KWM .eq. 777) then
        print *, 'pp043.F yes'
      else
        print *, 'pp043.F no: ', KWM
      end if
      end
