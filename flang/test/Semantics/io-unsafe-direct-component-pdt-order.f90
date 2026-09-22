! RUN: %python %S/test_errors.py %s %flang_fc1
!
! Regression test for the order-dependent visited-set bug in
! FindUnsafeIoDirectComponent (flang/lib/Semantics/check-io.cpp).  A derived
! type used in unformatted I/O may not have an allocatable or pointer direct
! component unless it is processed by defined I/O.  The check memoizes its walk;
! keying that memo on the shared type symbol (never erased on unwind) let a
! shielded parameterized-derived-type instantiation prune an unshielded sibling
! instantiation, silently suppressing the error.  The walk is now memoized on
! the instantiated scope, so the error surfaces regardless of traversal order.

module unsafe_pdt_order_mod
  type :: leaf(k)
    integer, kind :: k = 2
    real, allocatable :: a(:)          ! the unsafe direct component
  end type

  type :: branch(k)
    integer, kind :: k = 2
    type(leaf(k)) :: item
  end type

  ! Defined unformatted output for leaf(1) ONLY, so branch(1) is shielded and
  ! branch(2) is not.
  interface write(unformatted)
    module procedure wleaf1
  end interface

  ! Scope iterates components in SourceName (alphabetical) order, not
  ! declaration order.  The three containers pin down both traversal orders so
  ! the test does not silently depend on the component names chosen:
  !   - In `container`, the shielded branch(1) sorts first (a_safe < b_bad).
  !   - In `container_rev`, the unshielded branch(2) sorts first (a_bad <
  !     b_safe).
  !   - In `container_rev_decl`, the failing branch(2) is declared first but
  !     sorts last (a_safe < b_bad), so declaration order and traversal order
  !     disagree and the error must still surface.
  type :: container
    type(branch(1)) :: a_safe
    type(branch(2)) :: b_bad
  end type

  type :: container_rev
    type(branch(1)) :: b_safe
    type(branch(2)) :: a_bad
  end type

  type :: container_rev_decl
    type(branch(2)) :: b_bad
    type(branch(1)) :: a_safe
  end type

contains
  subroutine wleaf1(dtv, unit, iostat, iomsg)
    class(leaf(1)), intent(in) :: dtv
    integer, intent(in) :: unit
    integer, intent(out) :: iostat
    character(*), intent(in out) :: iomsg
    write(unit, iostat=iostat, iomsg=iomsg) size(dtv%a)
  end subroutine

  ! Positive control A: a lone shielded instantiation.  leaf(1) has matching
  ! defined unformatted output, so branch(1) is shielded and NO error is
  ! expected.
  subroutine test_shielded(u)
    integer, intent(in) :: u
    type(branch(1)) :: x
    write(u) x
  end subroutine

  ! Positive control B: a lone UNSHIELDED instantiation.  leaf(2) has no
  ! matching defined unformatted output, so its allocatable direct component is
  ! reached and the write is rejected.  This proves the branch(2)/leaf(2)
  ! subtree really is detectable on its own.
  subroutine test_unshielded(u)
    integer, intent(in) :: u
    type(branch(2)) :: y
    !ERROR: Derived type 'branch' in I/O cannot have an allocatable or pointer direct component 'a' unless using defined I/O
    write(u) y
  end subroutine

  ! The order-dependence: the shielded branch(1) is visited first, but the
  ! unshielded branch(2) must still be flagged.
  subroutine test_order_bug(u)
    integer, intent(in) :: u
    type(container) :: z
    !ERROR: Derived type 'container' in I/O cannot have an allocatable or pointer direct component 'a' unless using defined I/O
    write(u) z
  end subroutine

  ! Same as above, but the unshielded branch(2) is visited first in SourceName
  ! order.  The error must still be emitted regardless of traversal order.
  subroutine test_order_bug_rev(u)
    integer, intent(in) :: u
    type(container_rev) :: z
    !ERROR: Derived type 'container_rev' in I/O cannot have an allocatable or pointer direct component 'a' unless using defined I/O
    write(u) z
  end subroutine

  ! Failing branch(2) is declared first but sorts last (a_safe < b_bad), so it
  ! is visited last; the error must still be emitted regardless of the mismatch
  ! between declaration order and traversal order.
  subroutine test_order_bug_rev_decl(u)
    integer, intent(in) :: u
    type(container_rev_decl) :: z
    !ERROR: Derived type 'container_rev_decl' in I/O cannot have an allocatable or pointer direct component 'a' unless using defined I/O
    write(u) z
  end subroutine
end module
