! RUN: %python %S/test_errors.py %s %flang_fc1 -fenumeration-type
!
! This test verifies that the transversal order for enumeration type components
! does not impact the correct recognition/reporting of unformatted output
! errors.

module enum_pdt_order_mod
  !WARNING: ENUMERATION TYPE support is incomplete and should be enabled only for testing
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type

  type :: leaf(k)
    integer, kind :: k = 2
    type(color) :: c
  end type

  type :: branch(k)
    integer, kind :: k = 2
    type(leaf(k)) :: item
  end type

  ! Defined unformatted output for leaf(1) ONLY.
  interface write(unformatted)
    module procedure wleaf1
  end interface

  ! NOTE: Scope iterates components in SourceName (alphabetical) order, not
  ! declaration order.  The three containers below deliberately pin down both
  ! traversal orders so the test does not silently depend on the component
  ! names chosen:
  !   - In `container`, the shielded branch(1) sorts first (a_safe < b_bad).
  !   - In `container_rev`, the unshielded branch(2) sorts first (a_bad <
  !     b_safe).
  !   - In `container_rev_decl`, the failing branch(2) is declared first but
  !     sorts last (a_safe < b_bad), so declaration order and traversal order
  !     disagree and the error must still surface.
  ! Renaming a component in only one container would change which subtree is
  ! visited first; keeping both spellings covers the error path regardless of
  ! iteration order.  In `container_rev` the components are also declared in
  ! non-alphabetical order (b_safe before a_bad) so the source itself shows
  ! that SourceName order, not declaration order, drives the traversal.
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
    integer :: tmp
    ! Do not write dtv%c directly; an enumeration value may not appear in
    ! unformatted I/O.  Write a surrogate integer instead.
    tmp = int(dtv%c)
    write(unit, iostat=iostat, iomsg=iomsg) tmp
  end subroutine

  ! Positive control A: a lone shielded instantiation.  leaf(1) has matching
  ! defined unformatted output, so branch(1) expands to a single shielded item
  ! and NO error is expected.
  subroutine test_shielded(u)
    integer, intent(in) :: u
    type(branch(1)) :: x
    write(u) x
  end subroutine

  ! Positive control B: a lone UNSHIELDED instantiation.  leaf(2) has no
  ! matching defined unformatted output, so its enumeration component is
  ! reached and the write is rejected.  This proves the branch(2)/leaf(2)
  ! subtree really is detectable on its own.
  subroutine test_unshielded(u)
    integer, intent(in) :: u
    type(branch(2)) :: y
    !ERROR: Enumeration type is not supported in unformatted I/O
    write(u) y
  end subroutine

  ! Expected (correct, post-fix) behavior: the error below is emitted.
  subroutine test_order_bug(u)
    integer, intent(in) :: u
    type(container) :: z
    !ERROR: Enumeration type is not supported in unformatted I/O
    write(u) z
  end subroutine

  ! Same as above, but the unshielded branch(2) is visited first in SourceName
  ! order.  The error must still be emitted regardless of traversal order.
  subroutine test_order_bug_rev(u)
    integer, intent(in) :: u
    type(container_rev) :: z
    !ERROR: Enumeration type is not supported in unformatted I/O
    write(u) z
  end subroutine

  ! Failing branch(2) is declared first but sorts last (a_safe < b_bad), so it
  ! is visited last; the error must still be emitted regardless of the mismatch
  ! between declaration order and traversal order.
  subroutine test_order_bug_rev_decl(u)
    integer, intent(in) :: u
    type(container_rev_decl) :: z
    !ERROR: Enumeration type is not supported in unformatted I/O
    write(u) z
  end subroutine
end module
