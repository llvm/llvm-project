! RUN: %flang_fc1 -fdebug-dump-symbols %s 2>&1 | FileCheck %s
! CHECK:  Implied DO index 'j' uses an object of the same name in its bounds expressions
! CHECK: ObjectEntity type: REAL(4) shape: 1_8:5_8 init:[REAL(4)::1._4,2._4,3._4,4._4,5._4]
! Verify that the scope of a DATA statement implied DO loop index does
! not include the bounds expressions (language extension, with warning)
integer, parameter :: j = 5
real, save :: a(j)
data (a(j),j=1,j)/1,2,3,4,5/
end
