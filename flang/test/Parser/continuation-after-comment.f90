! RUN: %flang_fc1 -E %s 2>&1 | FileCheck %s

!Checking comments appearing in between continuation lines.
integer &
i(1)
/* \\\*/
!CHECK: integer i(1)
i(1)&
/* \\\*/
=1
!CHECK: i(1) =1
/* \*/
if (i(1)/=1)write(6,*)'error',i(1)
print *,'pass'
end
