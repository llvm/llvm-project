! RUN: %python %S/test_errors.py %s %flang_fc1 -fenumeration-type
! Test relational operators and SELECT CASE for enumeration types (F2023 7.6.2)

module enum_mod
  !WARNING: ENUMERATION TYPE support is incomplete and should be enabled only for testing
  enumeration type :: color
    enumerator :: red, green, blue
  end enumeration type

  !WARNING: ENUMERATION TYPE support is incomplete and should be enabled only for testing
  enumeration type :: direction
    enumerator :: north, south, east, west
  end enumeration type

  !WARNING: ENUMERATION TYPE support is incomplete and should be enabled only for testing
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

subroutine test_select_case_nested_integer(w, j)
  use enum_mod
  type(w_value), intent(in) :: w
  integer, intent(in) :: j

  ! Valid: an ordinary integer SELECT CASE nested in an arm of an
  ! enumeration SELECT CASE must not be checked against the enum type.
  select case (w)
    case (w1)
      select case (j)
        case (1)
          print *, 'one'
        case (2:4)
          print *, 'few'
        case default
          print *, 'many'
      end select
    case (w2)
      print *, 'w2'
  end select
end subroutine

subroutine test_select_case_nested_same_enum(w)
  use enum_mod
  type(w_value), intent(in) :: w

  ! Valid: a nested SELECT CASE over the same enumeration type.
  select case (w)
    case (w1)
      select case (w)
        case (w2)
          print *, 'inner w2'
        case (w3:w5)
          print *, 'inner range'
      end select
    case (w2)
      print *, 'w2'
  end select
end subroutine

subroutine test_select_case_nested_different_enum(w, c)
  use enum_mod
  type(w_value), intent(in) :: w
  type(color), intent(in) :: c

  ! Valid: a nested SELECT CASE over a different enumeration type.
  select case (w)
    case (w1)
      select case (c)
        case (red)
          print *, 'red'
        case (green)
          print *, 'green'
      end select
    case (w2)
      print *, 'w2'
  end select
end subroutine

subroutine test_select_case_nested_under_do_if(w, j, n)
  use enum_mod
  type(w_value), intent(in) :: w
  integer, intent(in) :: j, n
  integer :: i

  ! Valid: nested integer SELECT CASE buried under DO/IF in an enum arm.
  select case (w)
    case (w1)
      do i = 1, n
        if (j > 0) then
          select case (j)
            case (1)
              print *, 'one'
            case default
              print *, 'other'
          end select
        end if
      end do
    case (w2)
      print *, 'w2'
  end select
end subroutine

