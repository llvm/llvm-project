! RUN: %python %S/test_folding.py %s %flang_fc1
! Tests folding of conditional arguments (F2023 R1526-R1528)
! When a condition in a conditional-arg is a compile-time constant,
! the conditional-arg should be folded to the selected consequent,
! enabling further constant folding of the enclosing expression.

module m_funcs
  implicit none
  interface
    pure integer function func_int(x)
      integer, intent(in) :: x
    end function
    pure real function func_real(x)
      real, intent(in) :: x
    end function
    pure integer function func_two(x, y)
      integer, intent(in) :: x, y
    end function
    subroutine sub_optional(x)
      integer, intent(in), optional :: x
    end subroutine
  end interface
end module

module m
  use m_funcs
  implicit none

  ! Basic: .TRUE. selects the first consequent.
  logical, parameter :: test_true_int = abs((.true. ? -5 : -3)) == 5
  logical, parameter :: test_true_real = abs((.true. ? -2.0 : -4.0)) == 2.0

  ! Basic: .FALSE. selects the second consequent (tail).
  logical, parameter :: test_false_int = abs((.false. ? -5 : -3)) == 3
  logical, parameter :: test_false_real = abs((.false. ? -2.0 : -4.0)) == 4.0

  ! Multi-branch: first condition .TRUE. selects first consequent.
  logical, parameter :: test_multi_first = &
      abs((.true. ? -10 : .false. ? -20 : -30)) == 10

  ! Multi-branch: first .FALSE., second .TRUE. selects second consequent.
  logical, parameter :: test_multi_second = &
      abs((.false. ? -10 : .true. ? -20 : -30)) == 20

  ! Multi-branch: all conditions .FALSE. selects final consequent.
  logical, parameter :: test_multi_third = &
      abs((.false. ? -10 : .false. ? -20 : -30)) == 30

  ! Named constant as condition.
  logical, parameter :: cond_t = .true.
  logical, parameter :: cond_f = .false.
  logical, parameter :: test_named_cond_true = abs((cond_t ? -7 : -9)) == 7
  logical, parameter :: test_named_cond_false = abs((cond_f ? -7 : -9)) == 9

  ! Consequent expressions themselves are folded.
  integer, parameter :: k = 3
  logical, parameter :: test_consequent_fold = abs((.true. ? -(k*k) : -1)) == 9

  ! .TRUE. selecting a simple arithmetic consequent.
  logical, parameter :: test_true_arith = abs((.true. ? -(1+2) : -99)) == 3

  ! .FALSE. selecting a foldable tail expression.
  logical, parameter :: test_false_arith = abs((.false. ? -99 : -(2*3))) == 6

  ! Both consequents are foldable; .TRUE. picks the first.
  logical, parameter :: test_both_foldable_true = &
      abs((.true. ? -(1+2) : -(4+5))) == 3

  ! Both consequents are foldable; .FALSE. picks the second.
  logical, parameter :: test_both_foldable_false = &
      abs((.false. ? -(1+2) : -(4+5))) == 9

  ! Multi-branch with foldable consequent expressions.
  logical, parameter :: test_multi_foldable = &
      abs((.false. ? -(1+1) : .true. ? -(2+2) : -(3+3))) == 4

  ! Character consequent folding.
  logical, parameter :: test_char_true = len_trim((.true. ? 'hello' : 'world')) == 5
  logical, parameter :: test_char_false = len_trim((.false. ? 'hi   ' : 'world')) == 5

  ! Named constants as both conditions and consequents.
  integer, parameter :: val_a = -11, val_b = -22
  logical, parameter :: test_named_cons_true = abs((cond_t ? val_a : val_b)) == 11
  logical, parameter :: test_named_cons_false = abs((cond_f ? val_a : val_b)) == 22
  logical, parameter :: test_named_all = abs((cond_t ? val_a : val_b)) /= 22

  ! Condition is a constant expression (not a literal .TRUE./.FALSE.).
  integer, parameter :: n = 5
  logical, parameter :: test_expr_cond_true = abs((n > 3 ? -10 : -20)) == 10
  logical, parameter :: test_expr_cond_false = abs((n < 3 ? -10 : -20)) == 20
  logical, parameter :: test_expr_cond_eq = abs((n == 5 ? -42 : -99)) == 42
  logical, parameter :: test_expr_cond_multi = &
      abs((n > 10 ? -1 : n > 3 ? -2 : -3)) == 2

  ! Negative cases: verify the WRONG branch is NOT selected.
  logical, parameter :: test_neg_true_not_tail = abs((.true. ? -5 : -3)) /= 3
  logical, parameter :: test_neg_false_not_first = abs((.false. ? -5 : -3)) /= 5
  logical, parameter :: test_neg_multi_not_third = &
      abs((.true. ? -10 : .false. ? -20 : -30)) /= 30
  logical, parameter :: test_neg_multi_not_first = &
      abs((.false. ? -10 : .true. ? -20 : -30)) /= 10
  logical, parameter :: test_neg_expr_cond = abs((n > 3 ? -10 : -20)) /= 20

  ! Double-parenthesized conditional: parsed as a ConditionalExpr (not a
  ! ConditionalArg), so intrinsic folding works through the expression tree.
  logical, parameter :: test_double_paren_true = abs(((.true. ? -7 : -3))) == 7
  logical, parameter :: test_double_paren_false = abs(((.false. ? -7 : -3))) == 3
  logical, parameter :: test_double_paren_multi = &
      abs(((.false. ? -10 : .true. ? -20 : -30))) == 20

contains

  ! Non-intrinsic (user-defined) function calls with conditional args.
  ! The conditional arg is folded when the condition is constant, and the
  ! resolved consequent is passed to the function.  The function call itself
  ! is not folded (only intrinsics are), but the conditional arg folding
  ! must not crash.

  subroutine test_user_func_const_cond
    integer :: r
    ! Constant condition: conditional arg folds to -5, passed to func_int.
    r = func_int((.true. ? -5 : -3))
    r = func_int((.false. ? -5 : -3))
  end subroutine

  subroutine test_user_func_real_const_cond
    real :: r
    r = func_real((.true. ? -2.0 : -4.0))
    r = func_real((.false. ? -2.0 : -4.0))
  end subroutine

  subroutine test_user_func_multi_branch
    integer :: r
    ! Multi-branch with constant conditions.
    r = func_int((.false. ? -10 : .true. ? -20 : -30))
  end subroutine

  subroutine test_user_func_named_const
    integer :: r
    logical, parameter :: flag = .true.
    integer, parameter :: a = -11, b = -22
    r = func_int((flag ? a : b))
  end subroutine

  subroutine test_user_func_two_args
    integer :: r
    ! One conditional arg and one plain arg.
    r = func_two((.true. ? -5 : -3), 10)
  end subroutine

  subroutine test_user_func_runtime_cond
    integer :: r, a, b
    logical :: flag
    ! Runtime condition: conditional arg is NOT folded, passed through as-is.
    r = func_int((flag ? a : b))
    r = func_two((flag ? a : b), a)
  end subroutine

  ! Chain-peeling: leading constant-.FALSE. conditions are peeled off,
  ! and constant conditions after the runtime one are folded.
  ! FoldConditionalArgImpl promotes the inner chain to replace the
  ! peeled-off prefix, so the runtime condition becomes the first.

  subroutine test_chain_peel_false_then_runtime(flag)
    logical, intent(in) :: flag
    integer :: a, b, c, d, r
    ! (.false. ? a : .false. ? b : flag ? c : d)
    !   → peels two .FALSE. prefixes → (flag ? c : d)
    r = func_int((.false. ? a : .false. ? b : flag ? c : d))
  end subroutine

  subroutine test_chain_peel_false_runtime_true(flag)
    logical, intent(in) :: flag
    integer :: a, b, c, d, e, r
    ! (.false. ? a : .false. ? b : flag ? c : .true. ? d : e)
    !   → peels two .FALSE. prefixes, folds trailing .TRUE. → (flag ? c : d)
    r = func_int((.false. ? a : .false. ? b : flag ? c : .true. ? d : e))
  end subroutine

  subroutine test_chain_peel_single_false(flag)
    logical, intent(in) :: flag
    integer :: a, b, c, r
    ! (.false. ? a : flag ? b : c)
    !   → peels one .FALSE. prefix → (flag ? b : c)
    r = func_int((.false. ? a : flag ? b : c))
  end subroutine

  subroutine test_chain_peel_deep(flag)
    logical, intent(in) :: flag
    integer :: a, b, c, d, e, f, g, r
    ! (.false. ? a : .false. ? b : .false. ? c : .false. ? d : flag ? e : .true. ? f : g)
    !   → peels four .FALSE. prefixes, folds trailing .TRUE. → (flag ? e : f)
    r = func_int(( &
        .false. ? a : .false. ? b : .false. ? c : .false. ? d : &
        flag ? e : .true. ? f : g))
  end subroutine

  subroutine test_chain_peel_false_then_true
    integer :: a, b, c, r
    ! All conditions constant — fully resolved (no runtime condition).
    ! (.false. ? a : .false. ? b : .true. ? c : a)
    !   → peels two .FALSE., then .TRUE. selects c → fully folded.
    ! This must not crash and the call receives a plain expression.
    r = func_int((.false. ? a : .false. ? b : .true. ? c : a))
  end subroutine

  ! .NIL. folding tests: conditional args with .NIL. and optional dummies.
  ! When the condition is constant, .NIL. branches fold to absent arguments
  ! and non-.NIL. branches fold to plain expressions.

  subroutine test_nil_false_selects_nil
    integer :: a
    ! (.FALSE. ? a : .NIL.) → folds to .NIL. (absent)
    call sub_optional((.false. ? a : .nil.))
  end subroutine

  subroutine test_nil_false_selects_value
    integer :: a
    ! (.FALSE. ? .NIL. : a) → folds to a
    call sub_optional((.false. ? .nil. : a))
  end subroutine

  subroutine test_nil_true_selects_nil
    integer :: a
    ! (.TRUE. ? .NIL. : a) → folds to .NIL. (absent)
    call sub_optional((.true. ? .nil. : a))
  end subroutine

  subroutine test_nil_true_selects_value
    integer :: a
    ! (.TRUE. ? a : .NIL.) → folds to a
    call sub_optional((.true. ? a : .nil.))
  end subroutine

  subroutine test_nil_runtime_no_fold(flag)
    logical, intent(in) :: flag
    integer :: a
    ! (flag ? a : .NIL.) → no fold (runtime condition)
    call sub_optional((flag ? a : .nil.))
  end subroutine

  subroutine test_nil_chain_folds_to_nil
    integer :: a, b
    ! (.FALSE. ? a : .TRUE. ? .NIL. : b) → chain folds to .NIL. (absent)
    call sub_optional((.false. ? a : .true. ? .nil. : b))
  end subroutine

  ! Logical-type consequents: conditional args whose consequent expressions
  ! are logical values.  Tests folding through the conditional arg path.

  subroutine test_logical_consequent_true
    logical :: a, b, r
    ! Constant condition with logical consequents.
    ! (.TRUE. ? a : b) folds to 'a', passed to a function.
    a = .true.
    b = .false.
    r = (.true. ? a : b)
  end subroutine

  subroutine test_logical_consequent_false
    logical :: a, b, r
    a = .true.
    b = .false.
    r = (.false. ? a : b)
  end subroutine

end module
