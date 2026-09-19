! RUN: %check_flang_tidy %s bugprone-short-circuit %t

! Should trigger warning - basic AND case
subroutine test_basic_and(a, result)
  integer, optional, intent(in) :: a
  integer, intent(out) :: result
  ! CHECK-MESSAGES: :[[@LINE+1]]:3: warning: optional argument 'a' used in logical expression alongside present()
  if (present(a) .and. a > 0) then
    result = a
  end if
end subroutine test_basic_and

! Should trigger warning - basic OR case
subroutine test_basic_or(a, result)
  integer, optional, intent(in) :: a
  integer, intent(out) :: result
  ! CHECK-MESSAGES: :[[@LINE+1]]:3: warning: optional argument 'a' used in logical expression alongside present()
  if (present(a) .or. a == 42) then
    result = a
  end if
end subroutine test_basic_or

! Should trigger warning - reversed order
subroutine test_reversed_order(b, result)
  integer, optional, intent(in) :: b
  integer, intent(out) :: result
  ! CHECK-MESSAGES: :[[@LINE+1]]:3: warning: optional argument 'b' used in logical expression alongside present()
  if (b < 100 .and. present(b)) then
    result = b
  end if
end subroutine test_reversed_order

! Should trigger warning - complex expression
subroutine test_complex_expression(x, result)
  integer, optional, intent(in) :: x
  integer, intent(out) :: result
  logical :: flag = .true.
  ! CHECK-MESSAGES: :[[@LINE+1]]:3: warning: optional argument 'x' used in logical expression alongside present()
  if (present(x) .and. flag .and. x > 10) then
    result = x
  end if
end subroutine test_complex_expression

! Should trigger warning - parenthesized expression
subroutine test_parentheses(z, result)
  integer, optional, intent(in) :: z
  integer, intent(out) :: result
  ! CHECK-MESSAGES: :[[@LINE+1]]:3: warning: optional argument 'z' used in logical expression alongside present()
  if (present(z) .and. (z > 0 .and. z < 100)) then
    result = z
  end if
end subroutine test_parentheses

! Should trigger warning - multiple optional args
subroutine test_multiple_optionals(a, b, c, result)
  integer, optional, intent(in) :: a, b, c
  integer, intent(out) :: result
  ! CHECK-MESSAGES: :[[@LINE+2]]:3: warning: optional argument 'b' used in logical expression alongside present()
  ! CHECK-MESSAGES: :[[@LINE+1]]:3: warning: optional argument 'c' used in logical expression alongside present()
  if (present(b) .and. b > 0 .and. present(c) .and. c < 100) then
    result = b + c
  end if
end subroutine test_multiple_optionals

! Should NOT trigger warning - proper nested structure
subroutine test_proper_nesting(a, result)
  integer, optional, intent(in) :: a
  integer, intent(out) :: result
  if (present(a)) then
    if (a > 0) then
      result = a
    end if
  end if
end subroutine test_proper_nesting

! Should NOT trigger warning - only present() call
subroutine test_only_present(a, result)
  integer, optional, intent(in) :: a
  integer, intent(out) :: result
  if (present(a)) then
    result = 1
  end if
end subroutine test_only_present

! Should NOT trigger warning - only optional variable usage
subroutine test_only_optional(a, result)
  integer, optional, intent(in) :: a
  integer, intent(out) :: result
  if (a > 0) then
    result = a
  end if
end subroutine test_only_optional

! Should NOT trigger warning - multiple present() calls only
subroutine test_multiple_present_only(a, b, c, result)
  integer, optional, intent(in) :: a, b, c
  integer, intent(out) :: result
  if (present(a) .and. present(b) .and. present(c)) then
    result = 1
  end if
end subroutine test_multiple_present_only
