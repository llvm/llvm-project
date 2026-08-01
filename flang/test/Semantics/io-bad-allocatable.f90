! RUN: %python %S/test_errors.py %s %flang_fc1
! Test for https://github.com/llvm/llvm-project/issues/213324
! The 'visited' set in FindUnsafeIoDirectComponent used to be global ("seen
! anywhere") rather than path-scoped.  When a PDT was first reached via a
! shielded branch (with user-defined I/O), its typeSymbol was inserted and
! never erased, so a later unshielded branch sharing the same typeSymbol was
! silently pruned, producing a missed diagnostic.
module m
  type :: leaf(k)
    integer, kind :: k = 2
    real, allocatable :: a(:) ! the unsafe direct component
  end type

  interface write(unformatted)
    module procedure wleaf1 ! matches leaf(1) only
  end interface

  type :: branch(k)
    integer, kind :: k = 2
    type(leaf(k)) :: item
  end type

  type :: container
    type(branch(1)) :: a_safe ! visited first: leaf(1) is shielded
    type(branch(2)) :: b_bad  ! previously pruned; leaf(2)'s allocatable missed
  end type

contains
  subroutine wleaf1(dtv, unit, iostat, iomsg)
    class(leaf(1)), intent(in) :: dtv
    integer, intent(in) :: unit
    integer, intent(out) :: iostat
    character(*), intent(in out) :: iomsg
    iostat = 0
  end subroutine

  subroutine bug_case(u)
    integer, intent(in) :: u
    type(container) :: z
    !ERROR: Derived type 'container' in I/O cannot have an allocatable or pointer direct component 'a' unless using defined I/O
    write(u) z
  end subroutine

  subroutine control_case(u)
    integer, intent(in) :: u
    type(branch(2)) :: y
    !ERROR: Derived type 'branch(k=2_4)' in I/O cannot have an allocatable or pointer direct component 'a' unless using defined I/O
    write(u) y
  end subroutine
end module
