! RUN: not %flang -E %s 2>&1 | FileCheck %s
! CHECK: error: bad character ('&') in Fortran token
! #define KWM &, use for continuation w/o pasting (ifort and nag seem to continue #define)
#define KWM &

      integer :: j
      j = 666
      j = j + KWM
111
      if (j .eq. 777) then
        print *, 'pp130.F90 yes'
      else
        print *, 'pp130.F90 no', j
      end if
      end
