! RUN: %python %S/test_errors.py %s %flang_fc1
! Check SELECT CASE with LOGICAL selectors of every kind and CASE values of
! kinds different from the selector. CASE values are compared as LOGICAL(1),
! independently of the selector's kind.

subroutine s1(l1, l2, l4, l8)
  logical(1) :: l1
  logical(2) :: l2
  logical(4) :: l4
  logical(8) :: l8
  logical(1), parameter :: t1 = .true.
  logical(2), parameter :: f2 = .false.
  logical(8), parameter :: t8 = .true.

  select case (l1)
  case (.true._1)
  case (.false._8)
  !ERROR: CASE (.true._1) conflicts with previous cases
  case (.true._4)
  !ERROR: CASE (.false._1) conflicts with previous cases
  case (f2)
  end select

  select case (l2)
  case (.true._2)
  case (.false._2)
  !ERROR: CASE (.true._1) conflicts with previous cases
  case (.true._8)
  !ERROR: CASE (.false._1) conflicts with previous cases
  case (.false._1)
  end select

  select case (l4)
  case (t1)
  case default
  !ERROR: CASE (.true._1) conflicts with previous cases
  case (.true._2)
  !ERROR: CASE (.true._1) conflicts with previous cases
  case (t8)
  end select

  select case (l8)
  case (.false._4)
  !ERROR: CASE (.false._1) conflicts with previous cases
  case (f2)
  case (.true._8)
  end select

  select case (l8)
  case (.true._2)
  case (.false._1)
  case default
  end select

  select case (l2)
  !ERROR: CASE range is not allowed for LOGICAL
  case (.false._8:)
  !ERROR: CASE value has type 'INTEGER(4)' which is not compatible with the SELECT CASE expression's type 'LOGICAL(2)'
  case (1)
  end select
end
