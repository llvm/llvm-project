! Ensure argument -fdisable-real-16 works as expected.

! RUN: not %flang_fc1 -emit-hlfir -fdisable-real-16 %s -o - 2>&1 | FileCheck %s --check-prefixes=F128-DISABLE,WARN

! WARN: warning: '-fdisable-real-16' may cause SELECTED_REAL_KIND to return inconsistent results [-Wdisable-real-16]
! F128-DISABLE: error: REAL(KIND=16) is not an enabled type for this target
! F128-DISABLE: error: COMPLEX(KIND=16) is not an enabled type for this target
subroutine test_r16()
  real(16) :: r0
  complex(16) :: c0
end subroutine
