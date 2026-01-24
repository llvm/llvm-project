! RUN: %flang_fc1 -fdebug-dump-symbols %s 2>&1 | FileCheck %s

module m
contains
  ! SIMPLE: should end up with both SIMPLE and PURE (PURE implied by SIMPLE)
  simple subroutine s_simple()
  end

  ! PURE: should remain PURE, and should not acquire SIMPLE
  pure subroutine s_pure()
  end
end module

! CHECK: s_simple, PUBLIC, PURE, SIMPLE
! CHECK: s_pure, PUBLIC, PURE
! CHECK-NOT: s_pure, PUBLIC, PURE, SIMPLE

