! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
subroutine s1()
  ! C701 (R701) The type-param-value for a kind type parameter shall be a
  ! constant expression.
  !
  ! C702 (R701) A colon shall not be used as a type-param-value except in the 
  ! declaration of an entity that has the POINTER or ALLOCATABLE attribute.
  !
  ! C704 (R703) In a declaration-type-spec, every type-param-value that is 
  ! not a colon or an asterisk shall be a specification expression.
  !   Section 10.1.11 defines specification expressions
  !
  integer, parameter :: constVal = 1
  integer :: nonConstVal = 1
!ERROR: Invalid specification expression: reference to local entity 'nonconstval'
  character(nonConstVal) :: colonString1
  character(len=20, kind=constVal + 1) :: constKindString
  character(len=:, kind=constVal + 1), pointer :: constKindString1
!ERROR: The type parameter LEN cannot be deferred without the POINTER or ALLOCATABLE attribute
  character(len=:, kind=constVal + 1) :: constKindString2
!ERROR: Must be a constant value
  character(len=20, kind=nonConstVal) :: nonConstKindString
!ERROR: The type parameter LEN cannot be deferred without the POINTER or ALLOCATABLE attribute
  character(len=:) :: deferredString
!ERROR: The type parameter LEN cannot be deferred without the POINTER or ALLOCATABLE attribute
  character(:) :: colonString2
  !OK because of the allocatable attribute
  character(:), allocatable :: colonString3

!ERROR: Must have INTEGER type, but is REAL(4)
  character(3.5) :: badParamValue

  type derived(typeKind, typeLen)
    integer, kind :: typeKind
    integer, len :: typeLen
    character(typeKind) :: kindValue
    character(typeLen) :: lenValue
  end type derived

  type (derived(constVal, 3)) :: constDerivedKind
!ERROR: Value of kind type parameter 'typekind' (nonconstval) must be a scalar INTEGER constant
!ERROR: Invalid specification expression: reference to local entity 'nonconstval'
  type (derived(nonConstVal, 3)) :: nonConstDerivedKind

  !OK because all type-params are constants
  type (derived(3, constVal)) :: constDerivedLen

!ERROR: Invalid specification expression: reference to local entity 'nonconstval'
  type (derived(3, nonConstVal)) :: nonConstDerivedLen
!ERROR: The value of type parameter 'typelen' cannot be deferred without the POINTER or ALLOCATABLE attribute
  type (derived(3, :)) :: colonDerivedLen
!ERROR: The value of type parameter 'typekind' cannot be deferred without the POINTER or ALLOCATABLE attribute
!ERROR: The value of type parameter 'typelen' cannot be deferred without the POINTER or ALLOCATABLE attribute
  type (derived( :, :)) :: colonDerivedLen1
  type (derived( :, :)), pointer :: colonDerivedLen2
  type (derived(4, :)), pointer :: colonDerivedLen3
end subroutine s1
Program d5
  Type string(maxlen)
    Integer,Kind :: maxlen
    Character(maxlen) :: value
  End Type
  Type(string(80)) line
  line%value = 'ok'
  Print *,Trim(line%value)
End Program
