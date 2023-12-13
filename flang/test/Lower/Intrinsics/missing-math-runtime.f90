! There is no quad math runtime available in lowering
! for now. Test that the TODO are emitted correctly.
! RUN: bbc -emit-fir %s -o /dev/null 2>&1 | FileCheck %s

 complex(16) :: a
 real(16) :: b
! CHECK: not yet implemented: no math runtime available for 'ABS(COMPLEX(KIND=16))'
 b = abs(a)
end

