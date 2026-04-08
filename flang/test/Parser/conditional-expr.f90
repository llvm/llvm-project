! RUN: %flang_fc1 -fdebug-unparse-no-sema %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fdebug-dump-parse-tree-no-sema %s 2>&1 | FileCheck %s -check-prefix=TREE

! Test parsing of conditional expressions (Fortran 2023 R1002)

! Simple two-branch conditional
subroutine simple_conditional(x, y, z)
  integer :: x, y, z
  ! CHECK-LABEL: simple_conditional
  ! CHECK: z = ( x>5 ? y : 10 )
  ! TREE: ConditionalExpr
  ! TREE-NEXT: Scalar -> Logical -> Expr
  ! TREE: Expr -> Designator -> DataRef -> Name = 'y'
  ! TREE: Expr -> LiteralConstant -> IntLiteralConstant = '10'
  z = (x > 5 ? y : 10)
end subroutine

! Three-branch conditional (multiple conditions)
subroutine multi_branch_conditional(x, y, z)
  integer :: x, y, z
  ! CHECK-LABEL: multi_branch_conditional
  ! CHECK: z = ( x>10 ? 100 : ( y<5 ? 50 : 0 ) )
  ! TREE: ConditionalExpr
  ! TREE-NEXT: Scalar -> Logical -> Expr
  ! TREE: Expr -> LiteralConstant -> IntLiteralConstant = '100'
  ! TREE: Expr -> ConditionalExpr
  ! TREE: Scalar -> Logical -> Expr
  ! TREE: Expr -> LiteralConstant -> IntLiteralConstant = '50'
  ! TREE: Expr -> LiteralConstant -> IntLiteralConstant = '0'
  z = (x > 10 ? 100 : y < 5 ? 50 : 0)
end subroutine

! Nested conditionals
subroutine nested_conditionals(x, y, w, z, flag1, flag2)
  integer :: x, y, w, z
  logical :: flag1, flag2
  ! CHECK-LABEL: nested_conditionals
  ! Nested in value position
  ! CHECK: z = ( flag1 ? ( x>y ? x : y ) : 0 )
  ! TREE: ConditionalExpr
  ! TREE-NEXT: Scalar -> Logical -> Expr
  ! TREE: Expr -> ConditionalExpr
  ! TREE: Scalar -> Logical -> Expr
  ! TREE: Expr -> Designator -> DataRef -> Name = 'x'
  ! TREE: Expr -> Designator -> DataRef -> Name = 'y'
  ! TREE: Expr -> LiteralConstant -> IntLiteralConstant = '0'
  z = (flag1 ? (x > y ? x : y) : 0)
  ! Nested in condition
  ! CHECK: z = ( ( x>5 ? flag1 : flag2 ) ? y : 10 )
  z = ((x > 5 ? flag1 : flag2) ? y : 10)
  ! Multiple nested
  ! CHECK: z = ( x>10 ? ( y>20 ? 1 : 2 ) : ( w>30 ? 3 : 4 ) )
  z = (x > 10 ? (y > 20 ? 1 : 2) : (w > 30 ? 3 : 4))
end subroutine

! Basic type conditionals
subroutine basic_types(x, a, b, c, flag1, str1)
  integer :: x
  real :: a, b, c
  logical :: flag1
  character(len=10) :: str1
  ! CHECK-LABEL: basic_types
  ! Real type
  ! CHECK: c = ( a>b ? a : b )
  c = (a > b ? a : b)
  ! Logical type
  ! CHECK: flag1 = ( x>5 ? .TRUE. : .FALSE. )
  flag1 = (x > 5 ? .true. : .false.)
  ! Character type
  ! CHECK: str1 = ( flag1 ? "HELLO" : "WORLD" )
  str1 = (flag1 ? "HELLO" : "WORLD")
end subroutine

! Complex expressions in conditions and branches
subroutine complex_expressions(x, y, z, flag1)
  integer :: x, y, z
  logical :: flag1
  ! CHECK-LABEL: complex_expressions
  ! Complex expressions in branches
  ! CHECK: z = ( x>y ? x*2+1 : y*3-2 )
  z = (x > y ? x*2+1 : y*3-2)
  ! Complex logical condition
  ! CHECK: z = ( x>5.AND.y<10 ? x+y : x-y )
  z = (x > 5 .and. y < 10 ? x+y : x-y)
  ! Logical NOT
  ! CHECK: z = ( .NOT.flag1 ? x : y )
  z = (.not. flag1 ? x : y)
  ! Comparison chains
  ! CHECK: z = ( x>5.AND.x<10 ? x : 0 )
  z = (x > 5 .and. x < 10 ? x : 0)
  ! Parenthesized expressions in branches
  ! CHECK: z = ( x>5 ? (y+z) : (y-z) )
  z = (x > 5 ? (y+z) : (y-z))
end subroutine

! Many-branch conditionals
subroutine many_branches(x, z)
  integer :: x, z
  ! CHECK-LABEL: many_branches
  ! Four branches
  ! CHECK: z = ( x>10 ? 100 : ( x>5 ? 50 : ( x>0 ? 10 : 0 ) ) )
  z = (x > 10 ? 100 : x > 5 ? 50 : x > 0 ? 10 : 0)
  ! Five branches
  ! CHECK: z = ( x>20 ? 1 : ( x>15 ? 2 : ( x>10 ? 3 : ( x>5 ? 4 : 5 ) ) ) )
  z = (x > 20 ? 1 : x > 15 ? 2 : x > 10 ? 3 : x > 5 ? 4 : 5)
end subroutine

! Conditionals with arrays and functions
subroutine arrays_and_functions(x, y, z, arr, flag1)
  integer :: x, y, z, arr(5)
  logical :: flag1
  ! CHECK-LABEL: arrays_and_functions
  ! Array element in conditional
  ! CHECK: z = ( arr(1)>arr(2) ? arr(1) : arr(2) )
  z = (arr(1) > arr(2) ? arr(1) : arr(2))
  ! Function calls in conditional
  ! CHECK: x = ( abs(y)>10 ? abs(y) : y )
  x = (abs(y) > 10 ? abs(y) : y)
  ! Array constructor elements
  ! CHECK: arr(1:3) = [( flag1 ? x : y ), ( .NOT.flag1 ? x : y ), ( x>y ? x : y )]
  arr(1:3) = [(flag1 ? x : y), (.not. flag1 ? x : y), (x > y ? x : y)]
end subroutine

! Literals in conditionals
subroutine literals(x, z, a, c)
  integer :: x, z
  real :: a, c
  ! CHECK-LABEL: literals
  ! Real literals
  ! CHECK: c = ( a>0.0 ? 1.5 : 2.5 )
  c = (a > 0.0 ? 1.5 : 2.5)
  ! Negative values
  ! CHECK: z = ( x<0 ? -1 : 1 )
  z = (x < 0 ? -1 : 1)
end subroutine

! Conditional in specification expression context
function spec_expr_conditional(n, flag) result(res)
  integer, intent(in) :: n
  logical, intent(in) :: flag
  integer :: res
  ! CHECK-LABEL: spec_expr_conditional
  ! CHECK: res = ( flag ? n*2 : n )
  res = (flag ? n*2 : n)
end function

! Conditional with different integer kinds
subroutine integer_kinds(cond)
  integer(kind=4) :: i4a, i4b, i4c
  integer(kind=8) :: i8a, i8b, i8c
  logical :: cond
  ! CHECK-LABEL: integer_kinds
  ! CHECK: i4c = ( cond ? i4a : i4b )
  i4c = (cond ? i4a : i4b)
  ! CHECK: i8c = ( cond ? i8a : i8b )
  i8c = (cond ? i8a : i8b)
end subroutine

! Conditional with different real kinds
subroutine real_kinds(cond)
  real(kind=4) :: r4a, r4b, r4c
  real(kind=8) :: r8a, r8b, r8c
  logical :: cond
  ! CHECK-LABEL: real_kinds
  ! CHECK: r4c = ( cond ? r4a : r4b )
  r4c = (cond ? r4a : r4b)
  ! CHECK: r8c = ( cond ? r8a : r8b )
  r8c = (cond ? r8a : r8b)
end subroutine

! Conditional in various statement contexts
subroutine statement_contexts(flag)
  integer :: x, y, arr(10)
  logical :: flag
  ! CHECK-LABEL: statement_contexts
  ! In array constructor
  ! CHECK: arr(1:3) = [1, ( flag ? x : y ), 3]
  arr(1:3) = [1, (flag ? x : y), 3]
  ! In if statement condition
  ! CHECK: IF (( flag ? x : y )>5) THEN
  if ((flag ? x : y) > 5) then
    x = 1
  end if
  ! In print statement
  ! CHECK: PRINT *, ( flag ? x : y )
  print *, (flag ? x : y)
  ! In assignment to array element
  ! CHECK: arr(5) = ( flag ? x : y )
  arr(5) = (flag ? x : y)
end subroutine

! Complex type conditionals
subroutine complex_type(flag)
  complex :: c1, c2, c3
  complex(kind=8) :: c8a, c8b, c8c
  logical :: flag
  ! CHECK-LABEL: complex_type
  ! CHECK: c3 = ( flag ? c1 : c2 )
  c3 = (flag ? c1 : c2)
  ! CHECK: c8c = ( flag ? c8a : c8b )
  c8c = (flag ? c8a : c8b)
  ! With complex literals
  ! CHECK: c3 = ( flag ? (1.0,2.0) : (3.0,4.0) )
  c3 = (flag ? (1.0, 2.0) : (3.0, 4.0))
end subroutine

! Array-valued conditionals (F2023 10.1.4)
subroutine array_valued(flag)
  integer :: arr1(5), arr2(5), arr3(5)
  real :: mat1(3,3), mat2(3,3), mat3(3,3)
  logical :: flag
  ! CHECK-LABEL: array_valued
  ! Whole array conditional
  ! CHECK: arr3 = ( flag ? arr1 : arr2 )
  ! TREE: ConditionalExpr
  ! TREE-NEXT: Scalar -> Logical -> Expr
  ! TREE: Expr -> Designator -> DataRef -> Name = 'arr1'
  ! TREE: Expr -> Designator -> DataRef -> Name = 'arr2'
  arr3 = (flag ? arr1 : arr2)
  ! Multidimensional array conditional
  ! CHECK: mat3 = ( flag ? mat1 : mat2 )
  mat3 = (flag ? mat1 : mat2)
  ! Array section conditional
  ! CHECK: arr3(1:3) = ( flag ? arr1(1:3) : arr2(1:3) )
  arr3(1:3) = (flag ? arr1(1:3) : arr2(1:3))
end subroutine

! Derived type conditionals
subroutine derived_types(flag)
  type :: point
    real :: x, y
  end type
  type(point) :: p1, p2, p3
  logical :: flag
  ! CHECK-LABEL: derived_types
  ! CHECK: p3 = ( flag ? p1 : p2 )
  p3 = (flag ? p1 : p2)
end subroutine

! Character with different lengths
subroutine character_lengths(flag)
  character(len=5) :: short1, short2
  character(len=10) :: medium1, medium2
  character(len=20) :: long_result
  logical :: flag
  ! CHECK-LABEL: character_lengths
  ! Same length characters
  ! CHECK: short1 = ( flag ? "HELLO" : "WORLD" )
  short1 = (flag ? "HELLO" : "WORLD")
  ! Different length literals (type conformance rules apply)
  ! CHECK: long_result = ( flag ? "SHORT" : "MUCH LONGER STRING" )
  long_result = (flag ? "SHORT" : "MUCH LONGER STRING")
  ! Mixed variables and literals
  ! CHECK: medium1 = ( flag ? short1 : medium2 )
  medium1 = (flag ? short1 : medium2)
end subroutine

! Verify that '?' inside string literals and comments does not interfere
! with conditional expression parsing.
subroutine question_mark_chars(flag, str1, str2, str3)
  logical :: flag
  character(len=20) :: str1, str2, str3
  ! CHECK-LABEL: question_mark_chars
  ! CHECK: str1 = "HELLO?"
  str1 = "HELLO?"
  ! CHECK: str2 = ( flag ? "YES?" : "NO?" )
  str2 = (flag ? "YES?" : "NO?")
  ! CHECK: str3 = "WHAT? WHY? HOW?"
  str3 = "WHAT? WHY? HOW?"  ! ? in a comment
  ! CHECK: str2 = ( flag ? "MAYBE?" : "NOPE" )
  str2 = (flag ? "MAYBE?" : "NOPE")  ! ? in a trailing comment
end subroutine

! Verify that '(' and ')' inside character literals are handled correctly by
! ConditionalExprLookahead.
subroutine paren_in_char_literal(c, i)
  character(*), intent(in) :: c
  integer, intent(out) :: i
  ! CHECK-LABEL: paren_in_char_literal
  ! CHECK: i = ( c==")" ? 1 : 2 )
  i = (c == ")" ? 1 : 2)
  ! CHECK: i = ( c=="(" ? 1 : 2 )
  i = (c == "(" ? 1 : 2)
  ! CHECK: i = ( c=="()" ? 1 : 2 )
  i = (c == "()" ? 1 : 2)
end subroutine
