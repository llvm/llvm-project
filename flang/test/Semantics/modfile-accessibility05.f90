! RUN: %python %S/test_errors.py %s %flang_fc1
! Tests F2023 8.6.1 p2: module-name access-ids when the same entity is
! accessible via more than one USE-associated module.
!   - all source modules PRIVATE -> entity is PRIVATE
!   - any source module PUBLIC   -> entity is PUBLIC (PUBLIC wins)
!   - not all source modules named in access-stmts -> module default applies
module base
  implicit none
  integer :: x = 10
  integer :: y = 20
end module

! Both re-export x and y from base (default PUBLIC).
module re_a
  use base
end module

module re_b
  use base
end module

! Both source modules marked PRIVATE: x and y are PRIVATE.
module both_private
  use re_a
  use re_b
  implicit none
  private re_a
  private re_b
end module

! One source module PRIVATE, one PUBLIC: PUBLIC wins; x and y are PUBLIC.
module mixed_access
  use re_a
  use re_b
  implicit none
  private re_a
  public re_b
end module

! Only re_a has an access-stmt; re_b does not.
! F2023 8.6.1 requires ALL source modules to be named for the rule to apply;
! x and y fall back to the module default (PUBLIC).
module partial_stmt
  use re_a
  use re_b
  implicit none
  private re_a
end module

! Case 4: merged generics from two modules with mixed access-stmts.
! The generic interface 'op' should be PUBLIC (because gen_re_b is PUBLIC).
module gen_base
  implicit none
  interface op
    module procedure op_int
  end interface
contains
  integer function op_int(a)
    integer, intent(in) :: a
    op_int = a
  end function
end module

module gen_re_a
  use gen_base
end module

module gen_re_b
  use gen_base
end module

module gen_mixed
  use gen_re_a
  use gen_re_b
  implicit none
  private gen_re_a
  public gen_re_b
end module

program test
  implicit none
  ! Case 1: both source modules PRIVATE -> entities are PRIVATE.
  block
    ! ERROR: 'x' is PRIVATE in 'both_private'
    use both_private, only: x
  end block
  block
    ! ERROR: 'y' is PRIVATE in 'both_private'
    use both_private, only: y
  end block
  ! Case 2: PUBLIC wins over PRIVATE -> x and y are accessible (no error).
  block
    use mixed_access, only: x, y
  end block
  ! Case 3: partial access-stmt -> module default PUBLIC applies (no error).
  block
    use partial_stmt, only: x, y
  end block
  ! Case 4: merged generic PUBLIC via gen_re_b -> accessible (no error).
  block
    use gen_mixed, only: op
  end block
end program
