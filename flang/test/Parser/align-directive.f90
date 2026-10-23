! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

! Until lowering honors the directive, semantics says that it is ignored.
! CHECK: warning: {{.*}}!DIR$ ALIGN{{.*}}not yet honored

! Both spellings of the alignment directive parse to the same node and are
! unparsed in the canonical form.

subroutine s(n)
  integer :: n
  !dir$ align(64) :: a, b
  ! CHECK: !DIR$ ALIGN(64) :: a, b
  !dir$ attributes align : 32 :: c
  ! CHECK: !DIR$ ALIGN(32) :: c
  !DIR$ ATTRIBUTES ALIGN:16::d
  ! CHECK: !DIR$ ALIGN(16) :: d
  !dir$ align(128) e
  ! CHECK: !DIR$ ALIGN(128) :: e
  real :: a(n), b(n), c, d(10), e
  ! A bare ALIGN, or a spelling that merely starts with it, is still an
  ! unrecognized directive and is passed through unchanged.
  !dir$ align
  ! CHECK: !DIR$ align
  !dir$ alignfoo
  ! CHECK: !DIR$ alignfoo
  ! !DEC$ lines are plain comments.
  !dec$ attributes align : 64 :: a
  ! CHECK-NOT: !DEC$
end subroutine
