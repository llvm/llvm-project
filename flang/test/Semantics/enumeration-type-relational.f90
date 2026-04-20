! RUN: %python %S/test_errors.py %s %flang_fc1
! Test relational operators and SELECT CASE for enumeration types (F2023 7.6.2)

module enum_mod
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type

  enumeration type :: direction
    enumerator :: north, south, east, west
  end enumeration type

  enumeration type :: w_value
    enumerator :: w1, w2, w3, w4, w5
  end enumeration type
end module

subroutine test_relational_same_type()
  use enum_mod
  logical :: result

  ! Valid: all six relational operators between same-type enumerators
  result = red == red
  result = red /= green
  result = red < green
  result = green > red
  result = red <= red
  result = blue >= green
end subroutine

subroutine test_relational_cross_type()
  use enum_mod

  ! ERROR: Operands of .EQ. must have comparable types; have TYPE(color) and TYPE(direction)
  if (red == north) stop 1

  ! ERROR: Operands of .LT. must have comparable types; have TYPE(color) and TYPE(direction)
  if (red < north) stop 2
end subroutine

subroutine test_relational_enum_vs_integer()
  use enum_mod

  ! ERROR: Operands of .EQ. must have comparable types; have TYPE(color) and INTEGER(4)
  if (red == 1) stop 1

  ! ERROR: Operands of .EQ. must have comparable types; have INTEGER(4) and TYPE(color)
  if (1 == red) stop 2
end subroutine

subroutine test_select_case_basic(w)
  use enum_mod
  type(w_value), intent(in) :: w

  ! Valid: SELECT CASE with enumerator names as case values
  select case (w)
    case (w1)
      print *, 'w1'
    case (w2)
      print *, 'w2'
    case default
      print *, 'other'
  end select
end subroutine

subroutine test_select_case_range(w)
  use enum_mod
  type(w_value), intent(in) :: w

  ! Valid: SELECT CASE with ranges
  select case (w)
    case (w1)
      print *, 'w1'
    case (w2:w4)
      print *, 'w2 to w4'
    case (w5)
      print *, 'w5'
  end select
end subroutine

subroutine test_select_case_wrong_enum(w)
  use enum_mod
  type(w_value), intent(in) :: w

  select case (w)
    !ERROR: CASE value has type 'color' which is not compatible with the SELECT CASE expression's type 'ENUMERATION TYPE :: w_value'
    case (red)
      print *, 'wrong'
    case default
      print *, 'ok'
  end select
end subroutine

subroutine test_select_case_integer_case(w)
  use enum_mod
  type(w_value), intent(in) :: w

  select case (w)
    !ERROR: CASE value has type 'INTEGER(4)' which is not compatible with the SELECT CASE expression's type 'ENUMERATION TYPE :: w_value'
    case (1)
      print *, 'wrong'
    case default
      print *, 'ok'
  end select
end subroutine

subroutine test_select_case_non_enum_derived()
  type :: my_type
    integer :: val
  end type
  type(my_type) :: x = my_type(1)

  !ERROR: SELECT CASE expression must be integer, logical, character, or enumeration type
  select case (x)
    case default
  end select
end subroutine
