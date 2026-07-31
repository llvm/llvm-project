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

  type :: container
    type(branch(1)) :: a_safe
    type(branch(2)) :: b_bad
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
    !ERROR: Enumeration type may not be used in unformatted I/O
    write(u) y
  end subroutine

  ! Expected (correct, post-fix) behavior: the error below is emitted.
  subroutine test_order_bug(u)
    integer, intent(in) :: u
    type(container) :: z
    !ERROR: Enumeration type may not be used in unformatted I/O
    write(u) z
  end subroutine
end module
