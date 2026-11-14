! RUN: %python %S/test_errors.py %s %flang_fc1 -funsigned

! Tests SELECT CASE with UNSIGNED selectors and values exceeding signed maximum.

subroutine s1(u1)
  unsigned(1), intent(in) :: u1
  select case (u1)
  case (0u_1:99u_1)
  case (200u_1:)
  end select
end subroutine

subroutine s2(u4)
  unsigned(4), intent(in) :: u4
  select case (u4)
  case (0u:99u)
  case (3000000000u:)
  end select
end subroutine

subroutine s_conflict(u1)
  unsigned(1), intent(in) :: u1
  select case (u1)
  case (150u_1:220u_1)
  !ERROR: CASE (200U_1:250U_1) conflicts with previous cases
  case (200u_1:250u_1)
  end select
end subroutine

subroutine s_inverted(u1)
  unsigned(1), intent(in) :: u1
  select case (u1)
  !WARNING: CASE has lower bound greater than upper bound [-Wempty-case]
  case (200u_1:99u_1)
  end select
end subroutine

subroutine s_straddle(u1)
  unsigned(1), intent(in) :: u1
  select case (u1)
  case (100u_1:200u_1)
  case (201u_1:)
  end select
end subroutine

subroutine s_conflict_straddle(u1)
  unsigned(1), intent(in) :: u1
  select case (u1)
  case (100u_1:200u_1)
  !ERROR: CASE (150U_1:250U_1) conflicts with previous cases
  case (150u_1:250u_1)
  end select
end subroutine

subroutine s_conflict_open(u1)
  unsigned(1), intent(in) :: u1
  select case (u1)
  case (:200u_1)
  !ERROR: CASE (100U_1:) conflicts with previous cases
  case (100u_1:)
  end select
end subroutine

subroutine s_conflict_k8(u8)
  unsigned(8), intent(in) :: u8
  select case (u8)
  case (9223372036854775808u_8:9223372036854775809u_8)
  !ERROR: CASE (1U_8:9223372036854775808U_8) conflicts with previous cases
  case (1u_8:9223372036854775808u_8)
  end select
end subroutine
