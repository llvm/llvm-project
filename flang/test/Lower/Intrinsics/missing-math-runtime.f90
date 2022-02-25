! There is no quad math runtime available in lowering
! for now. Test that the TODO are emitted correctly.
! RUN: bbc -emit-fir %s -o /dev/null 2>&1 | FileCheck %s

 complex(16) :: a
 real(16) :: b
! CHECK: TODO: no math runtime available for 'hypot(f128, f128)'
 b = abs(a)
end

