! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s

! Until lowering honors the directive, semantics says that it is ignored.
! CHECK: warning: {{.*}}!DIR$ ALIGN{{.*}}not yet honored

! Fixed-form source: the directive sentinel may be spelled !DIR$, CDIR$ or
! *DIR$ in column 1; both spellings unparse to the canonical form.

      subroutine s(n)
      integer n
!dir$ align(64) :: a, b
! CHECK: !DIR$ ALIGN(64) :: a, b
cdir$ attributes align : 32 :: c
! CHECK: !DIR$ ALIGN(32) :: c
*DIR$ ALIGN(16) d
! CHECK: !DIR$ ALIGN(16) :: d
      real a(n), b(n), c, d(10)
! A bare ALIGN stays an unrecognized directive.
!dir$ align
! CHECK: !DIR$ align
      end
