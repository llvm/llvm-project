! #define KWM &, use for continuation w/o pasting (ifort and nag seem to continue #define)
#define KWM &

      integer :: j
      j = 666
      j = j + KWM
111
      if (j .eq. 777) then
        print *, 'pp130.F90 pass'
      else
        print *, 'pp130.F90 FAIL', j
      end if
      end
