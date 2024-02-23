! There is no quad math runtime available in lowering
! for now. Test that the TODO are emitted correctly.
! FIXME: the lit config has to flip a feature flag so that
! the tests can use different checks depending on whether
! REAL(16) math support is enabled or not.
! XFAIL: *
! RUN: bbc -emit-fir %s -o /dev/null 2>&1 | FileCheck %s

 complex(16) :: a
 real(16) :: b
! CHECK: compiler is built without support for 'ABS(COMPLEX(KIND=16))'
 b = abs(a)
end

