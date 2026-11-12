! RUN: %python %S/test_errors.py %s %flang_fc1
! Check for semantic errors in tokenize() subroutine calls
! Based on Fortran 2023 standard requirements

program test_tokenize_errors
  implicit none
  
  ! Valid declarations for reference
  character(:), allocatable :: tokens(:), separator(:)
  integer, allocatable :: first(:), last(:)
  character(20) :: string
  character(5) :: set
  
  ! Invalid declarations for testing
  integer :: int_scalar, int_array(10)
  real :: real_scalar
  character(10) :: fixed_tokens(10), fixed_separator(10)
  integer :: fixed_first(10), fixed_last(10)
  character(10) :: string_array(5)
  character(5) :: set_array(5)
  character(len=10, kind=2) :: wide_string, wide_set
  character(len=10, kind=2), allocatable :: wide_tokens(:)
  real, allocatable :: real_first(:), real_last(:)
  logical, allocatable :: logical_array(:)
  integer, allocatable :: first_2d(:,:), last_2d(:,:)
  character(:), allocatable :: tokens_2d(:,:)
  
  type t_coarray
    integer, allocatable :: a(:)
  end type
  type(t_coarray) :: coa[*]
  integer, allocatable :: coindexed_first(:)[:]
  integer, allocatable :: coindexed_last(:)[:]
  character(:), allocatable :: coindexed_tokens(:)[:]
  
  !========================================================================
  ! Test Form 1: TOKENIZE(STRING, SET, TOKENS [, SEPARATOR])
  !========================================================================
  
  ! Valid call (reference)
  call tokenize("hello world", " ", tokens)
  call tokenize(string, set, tokens, separator)
  
  !========================================================================
  ! Form 1: Wrong types for STRING argument
  !========================================================================
  
  !ERROR: Actual argument for 'string=' has bad type 'INTEGER(4)'
  call tokenize(int_scalar, set, tokens)
  
  !ERROR: Actual argument for 'string=' has bad type 'REAL(4)'
  call tokenize(real_scalar, set, tokens)
  
  !========================================================================
  ! Form 1: Wrong rank for STRING (must be scalar)
  !========================================================================
  ! Fails
  !ERROR: 'string=' argument has unacceptable rank 1
  call tokenize(string_array, set, tokens)
  
  !========================================================================
  ! Form 1: Wrong types for SET argument
  !========================================================================
  
  !ERROR: Actual argument for 'set=' has bad type 'INTEGER(4)'
  call tokenize(string, int_scalar, tokens)
  
  !ERROR: Actual argument for 'set=' has bad type 'REAL(4)'
  call tokenize(string, real_scalar, tokens)
  
  !========================================================================
  ! Form 1: Wrong rank for SET (must be scalar)
  !========================================================================
  ! Fails
  !ERROR: 'set=' argument has unacceptable rank 1
  call tokenize(string, set_array, tokens)
  
  !========================================================================
  ! Form 1: Wrong types for TOKENS argument
  !========================================================================
  ! Fails
  !ERROR: Actual argument for 'tokens=' has bad type 'INTEGER(4)'
  call tokenize(string, set, int_array)
  ! Fails
  !ERROR: Actual argument for 'tokens=' has bad type 'REAL(4)'
  call tokenize(string, set, real_first)
  ! Fails
  !ERROR: Actual argument for 'tokens=' has bad type 'LOGICAL(4)'
  call tokenize(string, set, logical_array)
  
  !========================================================================
  ! Form 1: Wrong rank for TOKENS (must be rank-1 array)
  !========================================================================
  ! Fails
  !ERROR: 'tokens=' argument has unacceptable rank 0
  call tokenize(string, set, string)
  ! Fails
  !ERROR: 'tokens=' argument has unacceptable rank 2
  call tokenize(string, set, tokens_2d)
  
  !========================================================================
  ! Form 1: TOKENS must be allocatable
  !========================================================================
  
  !ERROR: 'tokens=' argument to 'tokenize' must be ALLOCATABLE
  call tokenize(string, set, fixed_tokens)
  
  !========================================================================
  ! Form 1: Wrong types for optional SEPARATOR argument
  !========================================================================
  ! Fails
  !ERROR: Actual argument for 'separator=' has bad type 'INTEGER(4)'
  call tokenize(string, set, tokens, int_array)
  ! Fails
  !ERROR: Actual argument for 'separator=' has bad type 'REAL(4)'
  call tokenize(string, set, tokens, real_first)
  
  !========================================================================
  ! Form 1: Wrong rank for SEPARATOR (must be rank-1 array)
  !========================================================================
  ! Fails
  !ERROR: 'separator=' argument has unacceptable rank 0
  call tokenize(string, set, tokens, set)
  
  !========================================================================
  ! Form 1: SEPARATOR must be allocatable
  !========================================================================
  
  !ERROR: 'separator=' argument to 'tokenize' must be ALLOCATABLE
  call tokenize(string, set, tokens, fixed_separator)
  
  !========================================================================
  ! Form 1: Character kind mismatches
  !========================================================================
  ! Fails
  ! wide_string (kind=2) becomes sameArg; set (kind=1) fails sameKind check
  !ERROR: Actual argument for 'set=' has bad type or kind 'CHARACTER(KIND=1,LEN=5_8)'
  call tokenize(wide_string, set, tokens)
  ! Fails
  !ERROR: Actual argument for 'set=' has bad type or kind 'CHARACTER(KIND=2,LEN=10_8)'
  call tokenize(string, wide_set, tokens)
  ! Fails
  !ERROR: Actual argument for 'tokens=' has bad type or kind 'CHARACTER(KIND=2,LEN=10_8)'
  call tokenize(string, set, wide_tokens)
  
  !========================================================================
  ! Test Form 2: TOKENIZE(STRING, SET, FIRST, LAST)
  !========================================================================
  
  ! Valid call (reference)
  call tokenize("hello world", " ", first, last)
  
  !========================================================================
  ! Form 2: Wrong types for STRING argument (same as Form 1)
  !========================================================================
  
  !ERROR: Actual argument for 'string=' has bad type 'INTEGER(4)'
  call tokenize(int_scalar, set, first, last)
  
  !========================================================================
  ! Form 2: Wrong types for SET argument (same as Form 1)
  !========================================================================
  
  !ERROR: Actual argument for 'set=' has bad type 'INTEGER(4)'
  call tokenize(string, int_scalar, first, last)
  
  !========================================================================
  ! Form 2: Wrong types for FIRST argument
  !========================================================================
  
  !ERROR: Actual argument for 'first=' has bad type 'REAL(4)'
  call tokenize(string, set, real_first, last)

  !ERROR: Actual argument for 'first=' has bad type 'LOGICAL(4)'
  call tokenize(string, set, logical_array, last)

  !========================================================================
  ! Form 2: Wrong rank for FIRST (must be rank-1 array)
  !========================================================================

  !ERROR: 'first=' argument has unacceptable rank 0
  call tokenize(string, set, int_scalar, last)

  !ERROR: 'first=' argument has unacceptable rank 2
  call tokenize(string, set, first_2d, last)

  !========================================================================
  ! Form 2: FIRST must be allocatable
  !========================================================================

  !ERROR: 'first=' argument to 'tokenize' must be ALLOCATABLE
  call tokenize(string, set, fixed_first, last)

  !========================================================================
  ! Form 2: Wrong types for LAST argument
  !========================================================================

  !ERROR: Actual argument for 'last=' has bad type 'REAL(4)'
  call tokenize(string, set, first, real_first)

  !========================================================================
  ! Form 2: Wrong rank for LAST (must be rank-1 array)
  !========================================================================

  !ERROR: 'last=' argument has unacceptable rank 0
  call tokenize(string, set, first, int_scalar)

  !ERROR: 'last=' argument has unacceptable rank 2
  call tokenize(string, set, first, last_2d)

  !========================================================================
  ! Form 2: LAST must be allocatable
  !========================================================================

  !ERROR: 'last=' argument to 'tokenize' must be ALLOCATABLE
  call tokenize(string, set, first, fixed_last)
  
  !========================================================================
  ! Argument count errors
  !========================================================================
  
  !ERROR: missing mandatory 'set=' argument
  call tokenize(string)
  ! Fails
  !ERROR: missing mandatory 'tokens=' argument
  call tokenize(string, set)
  
  !ERROR: too many actual arguments for intrinsic 'tokenize'
  call tokenize(string, set, tokens, separator, first)
  
  !========================================================================
  ! Coindexed object restrictions (if applicable)
  !========================================================================
  
  ! Note: Coarray tests depend on whether the standard allows coindexed
  ! objects for TOKENIZE. Uncomment if compiler version enforces this.
  
  ! !ERROR: 'first=' argument to 'tokenize' may not be a coindexed object
  ! call tokenize(string, set, coindexed_first[1], last)
  
  ! !ERROR: 'last=' argument to 'tokenize' may not be a coindexed object
  ! call tokenize(string, set, first, coindexed_last[1])
  
  ! !ERROR: 'tokens=' argument to 'tokenize' may not be a coindexed object
  ! call tokenize(string, set, coindexed_tokens[1])
  
  !========================================================================
  ! Keyword argument errors
  !========================================================================
  
  !ERROR: unknown keyword argument to intrinsic 'tokenize'
  call tokenize(string, set, tokens, invalid_keyword=separator)
  ! Fails
  !ERROR: Actual argument for 'tokens=' has bad type 'INTEGER(4)'
  call tokenize(string=string, set=set, tokens=first, separator=separator)
  
  !========================================================================
  ! Type/kind inconsistency between STRING, SET, TOKENS, SEPARATOR
  !========================================================================
  
  ! All character arguments must have the same kind (but can have different lengths)
  ! This is implicitly handled by SameCharNoLen in the intrinsic definition
  ! Fails
  !ERROR: Actual argument for 'set=' has bad type or kind 'CHARACTER(KIND=2,LEN=10_8)'
  call tokenize(string=string, set=wide_set, tokens=tokens)
  
end program test_tokenize_errors
