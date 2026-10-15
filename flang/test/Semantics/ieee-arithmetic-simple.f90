! RUN: %python %S/test_errors.py %s %flang_fc1
module m
contains
  pure subroutine get_round(r)
    use ieee_arithmetic, only: ieee_round_type, ieee_get_rounding_mode
    type(ieee_round_type), intent(out) :: r
    call ieee_get_rounding_mode(r)
  end subroutine
  pure subroutine set_round(r)
    use ieee_arithmetic, only: ieee_round_type, ieee_set_rounding_mode
    type(ieee_round_type), intent(in) :: r
    call ieee_set_rounding_mode(r)
  end subroutine
  pure subroutine get_uflow(g)
    use ieee_arithmetic, only: ieee_get_underflow_mode
    logical, intent(out) :: g
    call ieee_get_underflow_mode(g)
  end subroutine
  pure subroutine set_uflow(g)
    use ieee_arithmetic, only: ieee_set_underflow_mode
    logical, intent(in) :: g
    call ieee_set_underflow_mode(g)
  end subroutine
end module
