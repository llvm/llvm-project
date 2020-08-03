! RUN: %f18 -E %s 2>&1 | FileCheck %s
! CHECK: if(.true.)then
! CHECK: print*,'pp036.F no: ',.true.
* #define FALSE TRUE ...  .FALSE. -> .TRUE.
#define FALSE TRUE
      if (.FALSE.) then
        print *, 'pp036.F yes'
      else
        print *, 'pp036.F no: ', .FALSE.
      end if
      end
