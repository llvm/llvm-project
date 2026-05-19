! RUN: %python %S/test_errors.py %s %flang_fc1
! Test semantic analysis of conditional arguments (F2023:R1526-R1528)
! Constraints F2023:C1538-C1545

module m_conditional_arg
  implicit none

  interface
    subroutine sub_int(x)
      integer, intent(in) :: x
    end subroutine
    subroutine sub_real(x)
      real, intent(in) :: x
    end subroutine
    subroutine sub_int_out(x)
      integer, intent(out) :: x
    end subroutine
    subroutine sub_int_inout(x)
      integer, intent(inout) :: x
    end subroutine
    subroutine sub_optional(x)
      integer, intent(in), optional :: x
    end subroutine
    subroutine sub_not_optional(x)
      integer, intent(in) :: x
    end subroutine
    subroutine sub_alloc(x)
      integer, intent(inout), allocatable :: x
    end subroutine
    subroutine sub_pointer(x)
      integer, intent(inout), pointer :: x
    end subroutine
    subroutine sub_array(x)
      integer, intent(in) :: x(:)
    end subroutine
    subroutine sub_assumed_rank(x)
      integer, intent(in) :: x(..)
    end subroutine
    subroutine sub_coarray(x)
      integer, intent(inout) :: x[*]
    end subroutine
    subroutine sub_two(x, y)
      integer, intent(in) :: x, y
    end subroutine
    subroutine sub_array_fixed(x)
      integer, intent(in) :: x(5)
    end subroutine
    subroutine sub_array_two(x, y)
      integer, intent(in) :: x(:), y(:)
    end subroutine
    subroutine sub_matrix(x)
      integer, intent(in) :: x(:,:)
    end subroutine
    subroutine sub_array_optional(x)
      integer, intent(in), optional :: x(:)
    end subroutine
    subroutine sub_matrix_optional(x)
      integer, intent(in), optional :: x(:,:)
    end subroutine
    pure integer function func_int(x)
      integer, intent(in) :: x
    end function
    pure real function func_real(x)
      real, intent(in) :: x
    end function
    pure integer function func_two(x, y)
      integer, intent(in) :: x, y
    end function
  end interface

contains

  ! =========================================================================
  ! Positive tests: valid conditional arguments
  ! =========================================================================

  subroutine test_valid_simple
    integer :: a, b
    logical :: flag
    ! Simple two-branch conditional arg with explicit interface
    call sub_int((flag ? a : b))
  end subroutine

  subroutine test_valid_multi_branch
    integer :: a, b, c
    logical :: flag, flag2
    ! Multi-branch conditional arg
    call sub_int((flag ? a : flag2 ? b : c))
  end subroutine

  subroutine test_valid_nil_optional
    integer :: a
    logical :: flag
    ! .NIL. is allowed with optional dummy
    call sub_optional((flag ? a : .NIL.))
  end subroutine

  subroutine test_valid_nil_middle
    integer :: a, b
    logical :: flag, flag2
    ! .NIL. in middle branch with optional dummy
    call sub_optional((flag ? .NIL. : flag2 ? a : b))
  end subroutine

  subroutine test_valid_expressions
    integer :: a, b
    logical :: flag
    ! Expression consequent-args with INTENT(IN)
    call sub_int((flag ? a + 1 : b * 2))
  end subroutine

  subroutine test_valid_intent_out
    integer :: a, b
    logical :: flag
    ! Variable consequent-args with INTENT(OUT)
    call sub_int_out((flag ? a : b))
  end subroutine

  subroutine test_valid_intent_inout
    integer :: a, b
    logical :: flag
    ! Variable consequent-args with INTENT(INOUT)
    call sub_int_inout((flag ? a : b))
  end subroutine

  subroutine test_valid_two_args
    integer :: a, b, c, d
    logical :: flag, flag2
    ! Multiple arguments, one conditional
    call sub_two(a, (flag ? b : c))
  end subroutine

  ! =========================================================================
  ! Implicit interface: conditional arguments require an explicit interface.
  ! =========================================================================

  subroutine test_implicit_interface
    integer :: a, b
    logical :: flag
    !ERROR: Conditional argument requires an explicit interface
    call external_sub_no_interface((flag ? a : b))
  end subroutine

  ! =========================================================================
  ! F2023:R1526: The condition in a conditional argument must be a
  !             scalar-logical-expr.
  ! =========================================================================

  subroutine test_nonlogical_condition
    integer :: a, b, icond
    real :: rcond
    character :: ccond
    logical :: flag
    !ERROR: Must have LOGICAL type, but is INTEGER(4)
    call sub_int((icond ? a : b))
    !ERROR: Must have LOGICAL type, but is REAL(4)
    call sub_int((rcond ? a : b))
    !ERROR: Must have LOGICAL type, but is CHARACTER(KIND=1,LEN=1_8)
    call sub_int((ccond ? a : b))
    ! Valid: logical condition
    call sub_int((flag ? a : b))
  end subroutine

  subroutine test_nonscalar_condition
    integer :: a, b
    logical :: arr_flag(5)
    !ERROR: Must be a scalar value, but is a rank-1 array
    call sub_int((arr_flag ? a : b))
  end subroutine

  ! =========================================================================
  ! F2023:C1538: Each consequent-arg shall have the same declared type and
  !              kind type parameters.
  ! =========================================================================

  subroutine test_f2023_c1538_different_type
    integer :: a
    real :: r
    logical :: flag
    !ERROR: All consequent-args in a conditional argument must have the same type and kind; have INTEGER(4) and REAL(4)
    call sub_int((flag ? a : r))
  end subroutine

  subroutine test_f2023_c1538_different_kind
    integer(4) :: a
    integer(8) :: b
    logical :: flag
    !ERROR: All consequent-args in a conditional argument must have the same type and kind; have INTEGER(4) and INTEGER(8)
    call sub_int((flag ? a : b))
  end subroutine

  subroutine test_f2023_c1538_three_branch_mismatch
    integer :: a, b
    real :: r
    logical :: flag, flag2
    !ERROR: All consequent-args in a conditional argument must have the same type and kind; have INTEGER(4) and REAL(4)
    call sub_int((flag ? a : flag2 ? b : r))
  end subroutine

  ! =========================================================================
  ! F2023:C1539: Either all consequent-args shall have the same rank, or be
  !              assumed-rank.
  ! =========================================================================

  subroutine test_f2023_c1539_rank_mismatch
    integer :: a
    integer :: arr(5)
    logical :: flag
    !ERROR: All consequent-args in a conditional argument must have the same rank
    call sub_int((flag ? a : arr))
  end subroutine

  subroutine test_f2023_c1539_mixed_assumed_rank_and_fixed_rank(x)
    integer, intent(in) :: x(..)
    logical :: flag
    integer :: a
    ! Mixing assumed-rank and non-assumed-rank violates F2023:C1539
    !ERROR: All consequent-args in a conditional argument must have the same rank or all must be assumed-rank
    call sub_int((flag ? x : a))
  end subroutine

  ! =========================================================================
  ! F2023:C1540: At least one consequent shall be a consequent-arg.
  !              If the corresponding dummy argument is not optional, .NIL.
  !              shall not appear.
  ! =========================================================================

  subroutine test_f2023_c1540_all_nil
    logical :: flag
    !ERROR: At least one consequent in a conditional argument must not be .NIL.
    call sub_optional((flag ? .NIL. : .NIL.))
  end subroutine

  subroutine test_f2023_c1540_nil_non_optional
    integer :: a
    logical :: flag
    !ERROR: .NIL. in conditional argument associated with non-optional dummy argument 'x='
    call sub_not_optional((flag ? a : .NIL.))
  end subroutine

  ! =========================================================================
  ! F2023:C1541: If its corresponding dummy argument is INTENT(OUT) or
  !              INTENT(INOUT), each consequent-arg shall be a variable.
  ! =========================================================================

  subroutine test_f2023_c1541_intent_out_expr
    integer :: a
    logical :: flag
    !ERROR: Each consequent-arg in conditional argument associated with INTENT(OUT) dummy argument 'x=' must be a variable
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
    !ERROR: 'a+1_4' is not a variable or pointer
    call sub_int_out((flag ? a + 1 : a))
  end subroutine

  subroutine test_f2023_c1541_intent_inout_expr
    integer :: a
    logical :: flag
    !ERROR: Each consequent-arg in conditional argument associated with INTENT(IN OUT) dummy argument 'x=' must be a variable
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'x=' is not definable
    !ERROR: 'a+1_4' is not a variable or pointer
    call sub_int_inout((flag ? a + 1 : a))
  end subroutine

  ! =========================================================================
  ! F2023:C1542: If its corresponding dummy argument is allocatable, a
  !              pointer, or a coarray, the attributes of each
  !              consequent-arg shall satisfy the requirements of that dummy
  !              argument.
  ! =========================================================================

  subroutine test_f2023_c1542_non_allocatable
    integer :: a, b
    logical :: flag
    !ERROR: ALLOCATABLE dummy argument 'x=' must be associated with an ALLOCATABLE actual argument
    call sub_alloc((flag ? a : b))
  end subroutine

  subroutine test_f2023_c1542_non_pointer
    integer :: a, b
    logical :: flag
    !ERROR: Actual argument associated with POINTER dummy argument 'x=' must also be POINTER unless INTENT(IN)
    call sub_pointer((flag ? a : b))
  end subroutine

  subroutine test_f2023_c1542_non_coarray
    integer :: a, b
    logical :: flag
    ! Non-coarray consequent-args passed to a coarray dummy violates F2023:C1542
    !ERROR: Each consequent-arg in conditional argument associated with a coarray dummy argument 'x=' must be a coarray
    !ERROR: Actual argument associated with coarray dummy argument 'x=' must be a coarray
    call sub_coarray((flag ? a : b))
  end subroutine

  ! =========================================================================
  ! F2023:C1543: A consequent-arg shall not be assumed-rank unless its
  !              corresponding dummy argument is also assumed-rank.
  ! =========================================================================

  subroutine test_f2023_c1543_assumed_rank_to_non_assumed_rank_dummy(x, y)
    integer, intent(in) :: x(..), y(..)
    logical :: flag
    ! Both consequents are assumed-rank (satisfies F2023:C1539) but the
    ! dummy is not assumed-rank — violates F2023:C1543.
    !ERROR: Assumed-rank consequent-arg in conditional argument may only be associated with assumed-rank dummy argument 'x='
    !ERROR: Assumed-rank actual argument may not be associated with a dummy argument 'x=' that is not also assumed-rank
    call sub_int((flag ? x : y))
  end subroutine

  ! =========================================================================
  ! F2023:C1544: A consequent-arg that is an expr shall not be a variable.
  !
  ! This is a grammar disambiguation rule making the "expr" and "variable"
  ! alternatives of R1528 mutually exclusive.  It is automatically satisfied
  ! because the parser collapses both alternatives into a single Expr
  ! production; IsVariable() determines the distinction at the semantic
  ! level.  The tests below demonstrate that the distinction works correctly:
  ! variables are accepted where variables are required (C1541), expressions
  ! are accepted where expressions suffice (INTENT(IN)), and non-variable
  ! expressions are rejected where variables are required.
  ! =========================================================================

  subroutine test_f2023_c1544_variable_with_intent_in
    integer :: a, b
    logical :: flag
    ! A variable consequent-arg is valid with INTENT(IN): it matched the
    ! "variable" alternative of R1528, satisfying C1544.
    call sub_int((flag ? a : b))
  end subroutine

  subroutine test_f2023_c1544_expr_with_intent_in
    integer :: a, b
    logical :: flag
    ! A non-variable expression is valid with INTENT(IN): it matched the
    ! "expr" alternative of R1528 and is not a variable, satisfying C1544.
    call sub_int((flag ? a + 1 : b * 2))
  end subroutine

  subroutine test_f2023_c1544_variable_with_intent_out
    integer :: a, b
    logical :: flag
    ! A variable consequent-arg is valid with INTENT(OUT): it matched the
    ! "variable" alternative, satisfying both C1544 and C1541.
    call sub_int_out((flag ? a : b))
  end subroutine

  subroutine test_f2023_c1544_expr_with_intent_out
    integer :: a
    logical :: flag
    ! A non-variable expression with INTENT(OUT) satisfies C1544 (it is an
    ! expr and not a variable) but violates C1541 (INTENT(OUT) requires a
    ! variable).  The C1541 error demonstrates the disambiguation works.
    !ERROR: Each consequent-arg in conditional argument associated with INTENT(OUT) dummy argument 'x=' must be a variable
    !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
    !ERROR: 'a+1_4' is not a variable or pointer
    call sub_int_out((flag ? a + 1 : a))
  end subroutine

  subroutine test_f2023_c1544_mixed_variable_and_expr
    integer :: a
    logical :: flag
    ! One consequent-arg is a variable, the other is an expression.
    ! Both are valid with INTENT(IN): each independently satisfies C1544
    ! (variable matched "variable", expression matched "expr").
    call sub_int((flag ? a : a + 1))
  end subroutine

  ! =========================================================================
  ! Typeless consequent-args: BOZ literals and NULL() without MOLD=
  ! are not allowed as consequent-args because they have no type.
  ! =========================================================================

  subroutine test_typeless_boz_consequent
    integer :: a
    logical :: flag
    !ERROR: Typeless expression is not allowed as a consequent in a conditional argument
    call sub_int((flag ? a : z'FF'))
  end subroutine

  subroutine test_typeless_boz_first_consequent
    integer :: a
    logical :: flag
    !ERROR: Typeless expression is not allowed as a consequent in a conditional argument
    call sub_int((flag ? z'FF' : a))
  end subroutine

  subroutine test_typeless_boz_both_consequents
    logical :: flag
    !ERROR: Typeless expression is not allowed as a consequent in a conditional argument
    call sub_int((flag ? z'FF' : z'00'))
  end subroutine

  subroutine test_typeless_boz_multi_branch
    integer :: a, b
    logical :: flag, flag2
    !ERROR: Typeless expression is not allowed as a consequent in a conditional argument
    call sub_int((flag ? a : flag2 ? b'101' : b))
  end subroutine

  subroutine test_typeless_null_consequent
    integer :: a
    logical :: flag
    !ERROR: Typeless expression is not allowed as a consequent in a conditional argument
    !ERROR: NULL() may not be used as an expression in this context
    call sub_int((flag ? a : null()))
  end subroutine

  subroutine test_typeless_null_first_consequent
    integer :: a
    logical :: flag
    !ERROR: Typeless expression is not allowed as a consequent in a conditional argument
    !ERROR: NULL() may not be used as an expression in this context
    call sub_int((flag ? null() : a))
  end subroutine

  subroutine test_typeless_null_both_consequents
    logical :: flag
    !ERROR: Typeless expression is not allowed as a consequent in a conditional argument
    !ERROR: NULL() may not be used as an expression in this context
    call sub_optional((flag ? null() : null()))
  end subroutine

  ! =========================================================================
  ! F2023:C1545: In a reference to a generic procedure, each consequent-arg
  !              shall have the same corank, and if any consequent-arg has the
  !              ALLOCATABLE or POINTER attribute, each consequent-arg shall
  !              have that attribute.
  ! =========================================================================

  subroutine test_f2023_c1545_allocatable_inconsistency
    integer, allocatable :: a
    integer :: b
    logical :: flag
    !ERROR: If any consequent-arg in a conditional argument has the ALLOCATABLE attribute, each must have it
    !ERROR: ALLOCATABLE dummy argument 'x=' must be associated with an ALLOCATABLE actual argument
    call sub_alloc((flag ? a : b))
  end subroutine

  subroutine test_f2023_c1545_pointer_inconsistency
    integer, pointer :: p
    integer :: b
    logical :: flag
    !ERROR: If any consequent-arg in a conditional argument has the POINTER attribute, each must have it
    !ERROR: Actual argument associated with POINTER dummy argument 'x=' must also be POINTER unless INTENT(IN)
    call sub_pointer((flag ? p : b))
  end subroutine

  subroutine test_f2023_c1545_allocatable_consistent_valid
    integer, allocatable :: a, b
    logical :: flag
    ! Both consequent-args are allocatable -- no C1545 error
    call sub_alloc((flag ? a : b))
  end subroutine

  subroutine test_f2023_c1545_pointer_consistent_valid
    integer, pointer :: p, q
    logical :: flag
    ! Both consequent-args are pointer -- no C1545 error
    call sub_pointer((flag ? p : q))
  end subroutine

  ! =========================================================================
  ! Shape analysis: verify that conditional args with array consequent-args
  ! get their shape from the consequents, not from the scalar condition.
  ! Without the GetShapeHelper(ActualArgument) override, the Traverse base
  ! class would return the scalar shape of the condition via CombineAnyOf,
  ! causing spurious shape mismatches.
  ! =========================================================================

  subroutine test_shape_array_conditional_arg
    integer :: a(5), b(5)
    logical :: flag
    ! Array conditional arg passed to assumed-shape dummy — shape comes
    ! from the consequent-args (rank 1), not the scalar condition.
    call sub_array((flag ? a : b))
  end subroutine

  subroutine test_shape_array_fixed_extent
    integer :: a(5), b(5)
    logical :: flag
    ! Array conditional arg with fixed-extent dummy.
    call sub_array_fixed((flag ? a : b))
  end subroutine

  subroutine test_shape_matrix_conditional_arg
    integer :: m1(3,4), m2(3,4)
    logical :: flag
    ! Rank-2 array conditional arg.
    call sub_matrix((flag ? m1 : m2))
  end subroutine

  subroutine test_shape_two_array_args
    integer :: a(5), b(5), c(5), d(5)
    logical :: f1, f2
    ! Two independent array conditional args in the same call.
    call sub_array_two((f1 ? a : b), (f2 ? c : d))
  end subroutine

  subroutine test_shape_nil_with_array
    integer :: a(5), b(5)
    logical :: flag, flag2
    ! .NIL. mixed with rank-1 array consequent-args passed to an optional
    ! array dummy.  If the shape analysis fails to extract rank from the
    ! first non-.NIL. consequent, this will produce a rank mismatch error.
    call sub_array_optional((flag ? .NIL. : flag2 ? a : b))
  end subroutine

  subroutine test_shape_nil_scalar_to_array_dummy
    integer :: a, b
    logical :: flag, flag2
    ! .NIL. with scalar consequent-args passed to an optional array dummy.
    ! Shape analysis extracts rank 0 from the first non-.NIL. consequent,
    ! which mismatches the assumed-shape array dummy.
    !ERROR: Scalar actual argument may not be associated with assumed-shape dummy argument 'x='
    call sub_array_optional((flag ? .NIL. : flag2 ? a : b))
  end subroutine

  subroutine test_shape_nil_skipped_rank_mismatch
    integer :: a(5), b(5)
    logical :: flag, flag2
    ! .NIL. is the first consequent; non-.NIL. consequents are rank-1 arrays.
    ! Dummy is a rank-2 matrix.  If shape were extracted from .NIL. (no shape),
    ! this might silently pass.  Instead, shape comes from the rank-1 array,
    ! producing a rank mismatch — proving .NIL. was correctly skipped.
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 1
    call sub_matrix_optional((flag ? .NIL. : flag2 ? a : b))
  end subroutine

  subroutine test_shape_nil_forced_rank_match
    integer :: a(5)
    ! .TRUE. forces .NIL. to be selected — after folding, the arg becomes
    ! absent.  But shape analysis runs on the unfolded conditional arg and
    ! still checks the rank of the non-.NIL. consequent (a, rank 1) against
    ! the dummy (rank 1).  No error because ranks match.
    call sub_array_optional((.true. ? .NIL. : a))
  end subroutine

  subroutine test_shape_nil_forced_rank_mismatch
    integer :: a(5)
    ! .TRUE. forces .NIL. to be selected, but shape analysis still extracts
    ! rank from the non-.NIL. consequent (a, rank 1).  The dummy is rank 2,
    ! so this errors — proving shape is NOT extracted from .NIL. even when
    ! .NIL. is the branch that will be selected at compile time.
    !ERROR: Rank of dummy argument is 2, but actual argument has rank 1
    call sub_matrix_optional((.true. ? .NIL. : a))
  end subroutine

  subroutine test_shape_nil_with_array_optional
    integer :: a(5), b(5)
    logical :: flag, flag2
    ! .NIL. with array consequent-args and an optional scalar dummy.
    ! Shape comes from the first non-.NIL. consequent (a(1), scalar).
    call sub_optional((flag ? .NIL. : flag2 ? a(1) : b(1)))
  end subroutine

  subroutine test_shape_multi_branch_array
    integer :: a(5), b(5), c(5)
    logical :: f1, f2
    ! Multi-branch conditional arg with arrays — all same rank.
    call sub_array((f1 ? a : f2 ? b : c))
  end subroutine

  ! =========================================================================
  ! Intrinsic characterization: conditional args with intrinsic procedures.
  ! Verifies that intrinsic argument matching correctly extracts the type
  ! and kind from a conditional arg via GetArgExpr().
  ! =========================================================================

  subroutine test_intrinsic_abs_int
    integer :: a, b, r
    logical :: flag
    ! ABS with integer conditional arg — runtime condition exercises
    ! the GetArgExpr() path in intrinsic characterization.
    r = abs((flag ? a : b))
  end subroutine

  subroutine test_intrinsic_abs_real
    real :: a, b, r
    logical :: flag
    r = abs((flag ? a : b))
  end subroutine

  subroutine test_intrinsic_max_conditional
    integer :: a, b, c, r
    logical :: flag
    ! MAX with a mix of conditional and plain args.
    r = max((flag ? a : b), c)
  end subroutine

  subroutine test_intrinsic_min_multi_conditional
    integer :: a, b, c, d, r
    logical :: f1, f2
    ! MIN with multiple conditional args.
    r = min((f1 ? a : b), (f2 ? c : d))
  end subroutine

  subroutine test_intrinsic_len_trim_char
    character(10) :: s1, s2
    logical :: flag
    integer :: r
    r = len_trim((flag ? s1 : s2))
  end subroutine

  subroutine test_intrinsic_multi_branch
    integer :: a, b, c, r
    logical :: f1, f2
    ! Multi-branch conditional arg in intrinsic call.
    r = abs((f1 ? a : f2 ? b : c))
  end subroutine

  ! =========================================================================
  ! Coarray: conditional args with coarray consequent-args.
  ! Verifies that GetCorank (via GetArgExpr()) correctly extracts the
  ! corank from a conditional arg's consequent.
  ! =========================================================================

  subroutine test_valid_coarray_conditional(a, b, flag)
    integer, intent(inout) :: a[*], b[*]
    logical, intent(in) :: flag
    ! Both consequent-args are coarrays — matches coarray dummy.
    call sub_coarray((flag ? a : b))
  end subroutine

  subroutine test_valid_coarray_multi_branch(a, b, c, f1, f2)
    integer, intent(inout) :: a[*], b[*], c[*]
    logical, intent(in) :: f1, f2
    ! Multi-branch with all coarray consequent-args.
    call sub_coarray((f1 ? a : f2 ? b : c))
  end subroutine

  ! =========================================================================
  ! Double-parenthesized conditional: the extra parens cause the parser to
  ! treat it as a ConditionalExpr (expression-level) rather than a
  ! ConditionalArg (argument-level).  This exercises the expression folding
  ! path for intrinsics instead of the ConditionalArg path.
  ! =========================================================================

  subroutine test_double_paren_abs_int
    integer :: a, b, r
    logical :: flag
    r = abs(((flag ? a : b)))
  end subroutine

  subroutine test_double_paren_abs_real
    real :: a, b, r
    logical :: flag
    r = abs(((flag ? a : b)))
  end subroutine

  subroutine test_double_paren_max
    integer :: a, b, c, r
    logical :: flag
    r = max(((flag ? a : b)), c)
  end subroutine

  subroutine test_double_paren_multi_branch
    integer :: a, b, c, r
    logical :: f1, f2
    r = abs(((f1 ? a : f2 ? b : c)))
  end subroutine

  ! =========================================================================
  ! Non-intrinsic (user-defined) function calls with conditional args.
  ! Unlike intrinsics, these skip the intrinsic folding path entirely,
  ! so the ConditionalArg in ActualArgument::u_ is passed through as-is.
  ! =========================================================================

  subroutine test_user_func_int
    integer :: a, b, r
    logical :: flag
    r = func_int((flag ? a : b))
  end subroutine

  subroutine test_user_func_real
    real :: a, b, r
    logical :: flag
    r = func_real((flag ? a : b))
  end subroutine

  subroutine test_user_func_multi_branch
    integer :: a, b, c, r
    logical :: f1, f2
    r = func_int((f1 ? a : f2 ? b : c))
  end subroutine

  subroutine test_user_func_two_args
    integer :: a, b, c, d, r
    logical :: f1, f2
    ! One conditional arg and one plain arg.
    r = func_two((f1 ? a : b), c)
  end subroutine

  subroutine test_user_func_two_conditional
    integer :: a, b, c, d, r
    logical :: f1, f2
    ! Both arguments are conditional args.
    r = func_two((f1 ? a : b), (f2 ? c : d))
  end subroutine

end module

! ===========================================================================
! Test conditional arguments used across module boundaries via USE
! ===========================================================================

module m_cond_arg_provider
  implicit none

  interface
    subroutine ext_sub_int(x)
      integer, intent(in) :: x
    end subroutine
    subroutine ext_sub_optional(x)
      integer, intent(in), optional :: x
    end subroutine
    subroutine ext_sub_real(x)
      real, intent(in) :: x
    end subroutine
    subroutine ext_sub_int_out(x)
      integer, intent(out) :: x
    end subroutine
  end interface

contains

  subroutine provider_valid(a, b, flag)
    integer, intent(in) :: a, b
    logical, intent(in) :: flag
    call ext_sub_int((flag ? a : b))
  end subroutine

  subroutine provider_valid_optional(a, flag)
    integer, intent(in) :: a
    logical, intent(in) :: flag
    call ext_sub_optional((flag ? a : .NIL.))
  end subroutine

end module

subroutine test_use_module_conditional_arg
  use m_cond_arg_provider
  implicit none
  integer :: x, y
  logical :: cond
  ! Valid: use procedures from a module with conditional args
  call provider_valid(x, y, cond)
  call provider_valid_optional(x, cond)
  ! Valid: use module interfaces directly with conditional args
  call ext_sub_int((cond ? x : y))
  call ext_sub_optional((cond ? x : .NIL.))
end subroutine

subroutine test_use_module_type_mismatch
  use m_cond_arg_provider
  implicit none
  integer :: a
  real :: r
  logical :: flag
  !ERROR: All consequent-args in a conditional argument must have the same type and kind; have INTEGER(4) and REAL(4)
  call ext_sub_int((flag ? a : r))
end subroutine

subroutine test_use_module_boz
  use m_cond_arg_provider
  implicit none
  integer :: a
  logical :: flag
  !ERROR: Typeless expression is not allowed as a consequent in a conditional argument
  call ext_sub_int((flag ? a : z'FF'))
end subroutine

subroutine test_use_module_null
  use m_cond_arg_provider
  implicit none
  integer :: a
  logical :: flag
  !ERROR: Typeless expression is not allowed as a consequent in a conditional argument
  !ERROR: NULL() may not be used as an expression in this context
  call ext_sub_int((flag ? null() : a))
end subroutine

subroutine test_use_module_nil_non_optional
  use m_cond_arg_provider
  implicit none
  integer :: a
  logical :: flag
  !ERROR: .NIL. in conditional argument associated with non-optional dummy argument 'x='
  call ext_sub_int((flag ? a : .NIL.))
end subroutine

subroutine test_use_module_intent_out_expr
  use m_cond_arg_provider
  implicit none
  integer :: a
  logical :: flag
  !ERROR: Each consequent-arg in conditional argument associated with INTENT(OUT) dummy argument 'x=' must be a variable
  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'x=' is not definable
  !ERROR: 'a+1_4' is not a variable or pointer
  call ext_sub_int_out((flag ? a + 1 : a))
end subroutine
! =========================================================================
! Derived type tests
! =========================================================================

module m_derived_types
  implicit none
  type :: t1
    integer :: x
  end type
  type :: t2
    integer :: x
  end type
  type, extends(t1) :: t1_ext
    integer :: y
  end type
  interface
    subroutine sub_t1(x)
      import :: t1
      type(t1), intent(in) :: x
    end subroutine
    subroutine sub_class_t1(x)
      import :: t1
      class(t1), intent(in) :: x
    end subroutine
    subroutine sub_class_star(x)
      class(*), intent(in) :: x
    end subroutine
    subroutine sub_type_star(x)
      type(*), intent(in) :: x
    end subroutine
  end interface
end module

subroutine test_derived_same_type
  use m_derived_types
  implicit none
  type(t1) :: a, b
  logical :: flag
  ! Valid: same declared type
  call sub_t1((flag ? a : b))
end subroutine

subroutine test_derived_mismatched_type
  use m_derived_types
  implicit none
  type(t1) :: a
  type(t2) :: b
  logical :: flag
  !ERROR: All consequent-args in a conditional argument must be the same derived type; have t1 and t2
  call sub_t1((flag ? a : b))
end subroutine

! Per F2023 15.5.2.3p4, mixing CLASS(t) and TYPE(t) with the same t is valid —
! the conditional arg becomes polymorphic.
subroutine test_polymorphic_class_and_type
  use m_derived_types
  implicit none
  type(t1), allocatable :: a
  class(t1), allocatable :: b
  logical :: flag
  ! Valid: CLASS(t1) mixed with TYPE(t1), same declared type per C1538
  call sub_class_t1((flag ? b : a))
end subroutine

! CLASS(*) mixed with non-CLASS(*) is invalid — no declared type to compare.
subroutine test_unlimited_poly_mismatch
  use m_derived_types
  implicit none
  type(t1) :: a
  class(*), allocatable :: b
  logical :: flag
  !ERROR: All consequent-args in a conditional argument must have the same type and kind; have CLASS(*) and t1
  call sub_class_star((flag ? b : a))
end subroutine

! Both CLASS(*) is valid.
subroutine test_unlimited_poly_both
  use m_derived_types
  implicit none
  class(*), allocatable :: a, b
  logical :: flag
  ! Valid: both unlimited polymorphic
  call sub_class_star((flag ? a : b))
end subroutine

! TYPE(*) with TYPE(*) — both assumed type, should be valid.
! F2023 15.5.2.3-2: consequent-args are actual arguments, so TYPE(*) is permitted.
subroutine test_assumed_type_both(a, b)
  use m_derived_types
  implicit none
  type(*), intent(in) :: a, b
  logical :: flag
  ! Valid: both assumed type
  call sub_type_star((flag ? a : b))
end subroutine

! TYPE(*) with TYPE(t1) — one assumed type, one concrete type, should error.
subroutine test_assumed_type_mismatch(a)
  use m_derived_types
  implicit none
  type(*), intent(in) :: a
  type(t1) :: b
  logical :: flag
  !ERROR: All consequent-args in a conditional argument must have the same type and kind; have TYPE(*) and t1
  call sub_type_star((flag ? a : b))
end subroutine

! CLASS(t1) with CLASS(t1) — both polymorphic same declared type, should be valid.
subroutine test_class_same_type
  use m_derived_types
  implicit none
  class(t1), allocatable :: a, b
  logical :: flag
  ! Valid: both CLASS(t1), same declared type
  call sub_class_t1((flag ? a : b))
end subroutine

! CLASS(t1) with CLASS(t2) — both polymorphic different declared types, should error.
subroutine test_class_different_types
  use m_derived_types
  implicit none
  class(t1), allocatable :: a
  class(t2), allocatable :: b
  logical :: flag
  !ERROR: All consequent-args in a conditional argument must be the same derived type; have CLASS(t1) and CLASS(t2)
  call sub_class_star((flag ? a : b))
end subroutine
