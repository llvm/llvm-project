! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s | FileCheck --check-prefix=TREE %s

! Until lowering honors the directive, semantics says that it is ignored.
! CHECK: warning: {{.*}}!DIR$ ALIGN{{.*}}not yet honored

! Both spellings of the alignment directive parse to the same node and are
! unparsed in the canonical form.

subroutine s(n)
  integer :: n
  !dir$ align(64) :: a, b
  ! CHECK: !DIR$ ALIGN(64) :: a, b
  ! TREE: CompilerDirective -> Align
  ! TREE-NEXT: uint64_t = '64'
  ! TREE-NEXT: Name = 'a'
  ! TREE-NEXT: Name = 'b'
  !dir$ attributes align : 32 :: c
  ! CHECK: !DIR$ ALIGN(32) :: c
  ! TREE: CompilerDirective -> Align
  ! TREE-NEXT: uint64_t = '32'
  ! TREE-NEXT: Name = 'c'
  !DIR$ ATTRIBUTES ALIGN:16::d
  ! CHECK: !DIR$ ALIGN(16) :: d
  !dir$ align(128) e
  ! CHECK: !DIR$ ALIGN(128) :: e
  !dir$ align(64) :: f, &
  !dir$&  g
  ! CHECK: !DIR$ ALIGN(64) :: f, g
  real :: a(n), b(n), c, d(10), e, f(n), g(n)
  ! A bare ALIGN, or a spelling that merely starts with it, is still an
  ! unrecognized directive and is passed through unchanged.
  !dir$ align
  ! CHECK: !DIR$ align
  !dir$ alignfoo
  ! CHECK: !DIR$ alignfoo
  ! !DEC$ lines are plain comments: no directive of any kind comes out of
  ! them, under either sentinel.
  !dec$ attributes align : 64 :: a
  ! CHECK-NOT: !DIR$
  ! CHECK: END SUBROUTINE
end subroutine
