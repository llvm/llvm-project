! RUN: %flang_fc1 -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s -check-prefix=TREE

! Test parsing of conditional arguments (F2023 R1526-R1528)

subroutine test_conditional_arg
  implicit none
  integer :: x, a, b, c
  integer :: arr(5)
  real :: r1, r2
  logical :: flag, flag2

  ! Test 1: Simple two-branch conditional arg
  ! CHECK: CALL sub(( x>0 ? a : b ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE-NEXT: Scalar -> Logical -> Expr -> GT
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'b'
  call sub((x > 0 ? a : b))

  ! Test 2: Multi-branch conditional arg
  ! CHECK: CALL sub(( x>10 ? a : x>5 ? b : c ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE-NEXT: Scalar -> Logical -> Expr -> GT
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
  ! TREE-NEXT: ConditionalArgTail -> ConditionalArg
  ! TREE-NEXT: Scalar -> Logical -> Expr -> GT
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'b'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'c'
  call sub((x > 10 ? a : x > 5 ? b : c))

  ! Test 3: .NIL. in else position (absent optional)
  ! CHECK: CALL sub(( flag ? a : .NIL. ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE-NEXT: Scalar -> Logical -> Expr -> Designator -> DataRef -> Name = 'flag'
  ! TREE-NEXT: Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> ConditionalArgNil
  call sub((flag ? a : .NIL.))

  ! Test 4: .NIL. in middle branch
  ! CHECK: CALL sub(( flag ? .NIL. : flag2 ? a : .NIL. ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE-NEXT: Scalar -> Logical -> Expr -> Designator -> DataRef -> Name = 'flag'
  ! TREE-NEXT: Consequent -> ConditionalArgNil
  ! TREE-NEXT: ConditionalArgTail -> ConditionalArg
  ! TREE-NEXT: Scalar -> Logical -> Expr -> Designator -> DataRef -> Name = 'flag2'
  ! TREE-NEXT: Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> ConditionalArgNil
  call sub((flag ? .NIL. : flag2 ? a : .NIL.))

  ! Test 5: Keyword argument with conditional arg
  ! CHECK: CALL sub(arg=( flag ? a : b ))
  ! TREE: ActualArgSpec
  ! TREE-NEXT: Keyword -> Name = 'arg'
  ! TREE-NEXT: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'b'
  call sub(arg = (flag ? a : b))

  ! Test 6: Multiple arguments, one conditional
  ! CHECK: CALL sub(arg1=a, arg2=( flag ? b : c ))
  ! TREE: ActualArg -> Expr -> Designator -> DataRef -> Name = 'a'
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'b'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'c'
  call sub(arg1 = a, arg2 = (flag ? b : c))

  ! Test 7: Expression consequent-args
  ! CHECK: CALL sub(( x>0 ? a+1 : b*2 ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> Add
  ! TREE: ConditionalArgTail -> Consequent -> Expr -> Multiply
  call sub((x > 0 ? a+1 : b*2))

  ! Test 8: Real variable consequent-args
  ! CHECK: CALL sub(( r1>0.0 ? r1 : r2 ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Scalar -> Logical -> Expr -> GT
  ! TREE: RealLiteralConstant
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'r1'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'r2'
  call sub((r1 > 0.0 ? r1 : r2))

  ! Test 9: Logical condition with .AND.
  ! CHECK: CALL sub(( x>0.AND.flag ? a : b ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE-NEXT: Scalar -> Logical -> Expr -> AND
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'b'
  call sub((x > 0 .and. flag ? a : b))

  ! Test 10: Logical condition with .NOT.
  ! CHECK: CALL sub(( .NOT.flag ? a : b ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE-NEXT: Scalar -> Logical -> Expr -> NOT
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'b'
  call sub((.not. flag ? a : b))

  ! Test 11: Logical condition with .OR.
  ! CHECK: CALL sub(( flag.OR.flag2 ? a : b ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE-NEXT: Scalar -> Logical -> Expr -> OR
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'b'
  call sub((flag .or. flag2 ? a : b))

  ! Test 12: Four-branch conditional arg
  ! CHECK: CALL sub(( x>30 ? a : x>20 ? b : x>10 ? c : x ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE-NEXT: Scalar -> Logical -> Expr -> GT
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
  ! TREE-NEXT: ConditionalArgTail -> ConditionalArg
  ! TREE-NEXT: Scalar -> Logical -> Expr -> GT
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'b'
  ! TREE-NEXT: ConditionalArgTail -> ConditionalArg
  ! TREE-NEXT: Scalar -> Logical -> Expr -> GT
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'c'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'x'
  call sub((x > 30 ? a : x > 20 ? b : x > 10 ? c : x))

  ! Test 13: Array element consequent-args (parsed as FunctionReference pre-sema)
  ! CHECK: CALL sub(( x>0 ? arr(1) : arr(2) ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> FunctionReference -> Call
  ! TREE-NEXT: ProcedureDesignator -> Name = 'arr'
  ! TREE: ConditionalArgTail -> Consequent -> Expr -> FunctionReference -> Call
  ! TREE-NEXT: ProcedureDesignator -> Name = 'arr'
  call sub((x > 0 ? arr(1) : arr(2)))

  ! Test 14: Array element with .NIL.
  ! CHECK: CALL sub(( flag ? arr(3) : .NIL. ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> FunctionReference -> Call
  ! TREE: ConditionalArgTail -> Consequent -> ConditionalArgNil
  call sub((flag ? arr(3) : .NIL.))

  ! Test 15: Multiple conditional args in one call
  ! CHECK: CALL sub(( flag ? a : b ), ( flag2 ? c : x ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'b'
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'x'
  call sub((flag ? a : b), (flag2 ? c : x))

  ! Test 16: Two keyword conditional args
  ! CHECK: CALL sub(p=( flag ? a : b ), q=( x>0 ? c : x ))
  ! TREE: Keyword -> Name = 'p'
  ! TREE-NEXT: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'b'
  ! TREE: Keyword -> Name = 'q'
  ! TREE-NEXT: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'c'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'x'
  call sub(p = (flag ? a : b), q = (x > 0 ? c : x))

  ! Test 17: Conditional arg with complex condition (comparison chain)
  ! CHECK: CALL sub(( x>5.AND.x<20 ? a : b ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE-NEXT: Scalar -> Logical -> Expr -> AND
  ! TREE-NEXT: Expr -> GT
  ! TREE: Expr -> LT
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'b'
  call sub((x > 5 .and. x < 20 ? a : b))

  ! Test 18: Consequent that is a parenthesized expression
  ! CHECK: CALL sub(( x>0 ? (a+b) : (b-c) ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> Parentheses -> Expr -> Add
  ! TREE: ConditionalArgTail -> Consequent -> Expr -> Parentheses -> Expr -> Subtract
  call sub((x > 0 ? (a+b) : (b-c)))

  ! Test 19: .NIL. only in middle of three branches
  ! CHECK: CALL sub(( flag ? .NIL. : flag2 ? a : b ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE-NEXT: Scalar -> Logical -> Expr -> Designator -> DataRef -> Name = 'flag'
  ! TREE-NEXT: Consequent -> ConditionalArgNil
  ! TREE-NEXT: ConditionalArgTail -> ConditionalArg
  ! TREE-NEXT: Scalar -> Logical -> Expr -> Designator -> DataRef -> Name = 'flag2'
  ! TREE-NEXT: Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'b'
  call sub((flag ? .NIL. : flag2 ? a : b))

  ! Test 20: All-NIL except one branch
  ! CHECK: CALL sub(( flag ? .NIL. : flag2 ? .NIL. : a ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE-NEXT: Scalar -> Logical -> Expr -> Designator -> DataRef -> Name = 'flag'
  ! TREE-NEXT: Consequent -> ConditionalArgNil
  ! TREE-NEXT: ConditionalArgTail -> ConditionalArg
  ! TREE-NEXT: Scalar -> Logical -> Expr -> Designator -> DataRef -> Name = 'flag2'
  ! TREE-NEXT: Consequent -> ConditionalArgNil
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
  call sub((flag ? .NIL. : flag2 ? .NIL. : a))

end subroutine

! Test conditional arg in a function reference context
subroutine test_func_ref
  implicit none
  integer :: x, a, b, result
  logical :: flag

  ! Test 21: Conditional arg passed to a function
  ! CHECK: result = func(( flag ? a : b ))
  ! TREE: FunctionReference -> Call
  ! TREE-NEXT: ProcedureDesignator -> Name = 'func'
  ! TREE-NEXT: ActualArgSpec
  ! TREE-NEXT: ActualArg -> ConditionalArg
  ! TREE-NEXT: Scalar -> Logical -> Expr -> Designator -> DataRef -> Name = 'flag'
  ! TREE-NEXT: Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'b'
  result = func((flag ? a : b))

  ! Test 22: Conditional arg as one of multiple function args
  ! CHECK: result = func(a, ( flag ? b : x ))
  ! TREE: ActualArg -> Expr -> Designator -> DataRef -> Name = 'a'
  ! TREE: ActualArg -> ConditionalArg
  ! TREE-NEXT: Scalar -> Logical -> Expr -> Designator -> DataRef -> Name = 'flag'
  ! TREE-NEXT: Consequent -> Expr -> Designator -> DataRef -> Name = 'b'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'x'
  result = func(a, (flag ? b : x))

end subroutine

! Test conditional arg with typeless consequents (BOZ literals, NULL())
! These should parse successfully; semantic errors are caught later.
subroutine test_typeless_consequents
  implicit none
  integer :: a, b
  logical :: flag, flag2

  ! Test 23: BOZ literal (hex) as consequent
  ! CHECK: CALL sub(( flag ? a : z"ff" ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> LiteralConstant -> BOZLiteralConstant
  call sub((flag ? a : z'FF'))

  ! Test 24: BOZ literal (binary) as consequent
  ! CHECK: CALL sub(( flag ? b"101" : a ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> LiteralConstant -> BOZLiteralConstant
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
  call sub((flag ? b'101' : a))

  ! Test 25: BOZ literal (octal) as consequent
  ! CHECK: CALL sub(( flag ? a : o"77" ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> LiteralConstant -> BOZLiteralConstant
  call sub((flag ? a : o'77'))

  ! Test 26: BOZ in both branches
  ! CHECK: CALL sub(( flag ? z"ff" : z"00" ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> LiteralConstant -> BOZLiteralConstant
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> LiteralConstant -> BOZLiteralConstant
  call sub((flag ? z'FF' : z'00'))

  ! Test 27: BOZ in middle branch of multi-branch conditional
  ! CHECK: CALL sub(( flag ? a : flag2 ? b"101" : b ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
  ! TREE-NEXT: ConditionalArgTail -> ConditionalArg
  ! TREE: Consequent -> Expr -> LiteralConstant -> BOZLiteralConstant
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'b'
  call sub((flag ? a : flag2 ? b'101' : b))

  ! Test 28: NULL() as consequent
  ! CHECK: CALL sub(( flag ? a : null() ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> FunctionReference -> Call
  ! TREE: ProcedureDesignator -> Name = 'null'
  call sub((flag ? a : null()))

  ! Test 29: NULL() as first consequent
  ! CHECK: CALL sub(( flag ? null() : a ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> FunctionReference -> Call
  ! TREE: ProcedureDesignator -> Name = 'null'
  ! TREE: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
  call sub((flag ? null() : a))

  ! Test 30: NULL() in both branches
  ! CHECK: CALL sub(( flag ? null() : null() ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> FunctionReference -> Call
  ! TREE: ProcedureDesignator -> Name = 'null'
  ! TREE: ConditionalArgTail -> Consequent -> Expr -> FunctionReference -> Call
  ! TREE: ProcedureDesignator -> Name = 'null'
  call sub((flag ? null() : null()))

end subroutine

! Test conditional arg inside a module and accessed via USE
module m_condarg_parse
  implicit none
contains
  subroutine mod_sub_with_condarg(flag, a, b)
    logical, intent(in) :: flag
    integer, intent(in) :: a, b
    ! Test 31: Conditional arg inside a module procedure
    ! CHECK: CALL ext(( flag ? a : b ))
    ! TREE: ActualArg -> ConditionalArg
    ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
    ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'b'
    call ext((flag ? a : b))
  end subroutine

  subroutine mod_sub_with_nil(flag, a)
    logical, intent(in) :: flag
    integer, intent(in) :: a
    ! Test 32: .NIL. conditional arg inside a module procedure
    ! CHECK: CALL ext(( flag ? a : .NIL. ))
    ! TREE: ActualArg -> ConditionalArg
    ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
    ! TREE-NEXT: ConditionalArgTail -> Consequent -> ConditionalArgNil
    call ext((flag ? a : .NIL.))
  end subroutine

  subroutine mod_sub_with_boz(flag, a)
    logical, intent(in) :: flag
    integer, intent(in) :: a
    ! Test 33: BOZ conditional arg inside a module procedure
    ! CHECK: CALL ext(( flag ? a : z"ff" ))
    ! TREE: ActualArg -> ConditionalArg
    ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
    ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> LiteralConstant -> BOZLiteralConstant
    call ext((flag ? a : z'FF'))
  end subroutine

  subroutine mod_sub_with_null(flag, a)
    logical, intent(in) :: flag
    integer, intent(in) :: a
    ! Test 34: NULL() conditional arg inside a module procedure
    ! CHECK: CALL ext(( flag ? null() : a ))
    ! TREE: ActualArg -> ConditionalArg
    ! TREE: Consequent -> Expr -> FunctionReference -> Call
    ! TREE: ProcedureDesignator -> Name = 'null'
    ! TREE: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'a'
    call ext((flag ? null() : a))
  end subroutine
end module

subroutine test_use_module_condarg
  use m_condarg_parse
  implicit none
  integer :: x, y
  logical :: cond

  ! Test 35: Conditional arg using module procedure
  ! CHECK: CALL mod_sub_with_condarg(cond, x, y)
  ! TREE: CallStmt
  ! TREE-NEXT: Call
  ! TREE-NEXT: ProcedureDesignator -> Name = 'mod_sub_with_condarg'
  call mod_sub_with_condarg(cond, x, y)

  ! Test 36: Conditional arg at call site using procedures from USEd module
  ! CHECK: CALL ext(( cond ? x : y ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'x'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'y'
  call ext((cond ? x : y))

  ! Test 37: BOZ conditional arg at call site after USE
  ! CHECK: CALL ext(( cond ? x : z"ff" ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'x'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> LiteralConstant -> BOZLiteralConstant
  call ext((cond ? x : z'FF'))

  ! Test 38: NULL() conditional arg at call site after USE
  ! CHECK: CALL ext(( cond ? null() : x ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> FunctionReference -> Call
  ! TREE: ProcedureDesignator -> Name = 'null'
  ! TREE: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'x'
  call ext((cond ? null() : x))

  ! Test 39: Multi-branch conditional arg at call site after USE
  ! CHECK: CALL ext(( cond ? x : cond ? y : .NIL. ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'x'
  ! TREE-NEXT: ConditionalArgTail -> ConditionalArg
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'y'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> ConditionalArgNil
  call ext((cond ? x : cond ? y : .NIL.))

end subroutine

! Test conditional arg with various kinds
subroutine test_kinds
  implicit none
  integer(kind=4) :: i4a, i4b
  integer(kind=8) :: i8a, i8b
  real(kind=4)    :: r4a, r4b
  real(kind=8)    :: r8a, r8b
  logical         :: cond

  ! Test 40: Integer kind=4 consequent-args
  ! CHECK: CALL sub(( cond ? i4a : i4b ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'i4a'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'i4b'
  call sub((cond ? i4a : i4b))

  ! Test 41: Integer kind=8 consequent-args
  ! CHECK: CALL sub(( cond ? i8a : i8b ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'i8a'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'i8b'
  call sub((cond ? i8a : i8b))

  ! Test 42: Real kind=4 consequent-args
  ! CHECK: CALL sub(( cond ? r4a : r4b ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'r4a'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'r4b'
  call sub((cond ? r4a : r4b))

  ! Test 43: Real kind=8 consequent-args
  ! CHECK: CALL sub(( cond ? r8a : r8b ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 'r8a'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 'r8b'
  call sub((cond ? r8a : r8b))

end subroutine

! Test conditional arg with character variables
subroutine test_character
  implicit none
  character(len=10) :: s1, s2
  logical :: flag

  ! Test 44: Character variable consequent-args
  ! CHECK: CALL sub(( flag ? s1 : s2 ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 's1'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> Expr -> Designator -> DataRef -> Name = 's2'
  call sub((flag ? s1 : s2))

  ! Test 45: Character variable with .NIL.
  ! CHECK: CALL sub(( flag ? s1 : .NIL. ))
  ! TREE: ActualArg -> ConditionalArg
  ! TREE: Consequent -> Expr -> Designator -> DataRef -> Name = 's1'
  ! TREE-NEXT: ConditionalArgTail -> Consequent -> ConditionalArgNil
  call sub((flag ? s1 : .NIL.))

end subroutine
