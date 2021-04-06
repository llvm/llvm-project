! RUN: %S/test_errors.sh %s %t %f18
! Test constant folding of type parameter values both a base value and a
! parameter name are supplied.
! 
! Type parameters are described in 7.5.3 and constant expressions are described 
! in 10.1.12.  10.1.12, paragraph 4 defines whether a specification inquiry is 
! a constant expression.  Section 10.1.11, paragraph 3, item (2) states that a 
! type parameter inquiry is a specification inquiry.  

module m1
  type dtype(goodDefaultKind, badDefaultKind)
    integer, kind :: goodDefaultKind = 4
    integer, kind :: badDefaultKind = 343
    ! next field OK only if instantiated with a good value of goodDefaultKind
    !ERROR: KIND parameter value (99) of intrinsic type REAL did not resolve to a supported value
    real(goodDefaultKind) :: goodDefaultField
    ! next field OK only if instantiated with a good value of goodDefaultKind
    !ERROR: KIND parameter value (343) of intrinsic type REAL did not resolve to a supported value
    !ERROR: KIND parameter value (99) of intrinsic type REAL did not resolve to a supported value
    real(badDefaultKind) :: badDefaultField
  end type dtype
  type(dtype) :: v1
  type(dtype(4, 4)) :: v2
  type(dtype(99, 4)) :: v3
  type(dtype(4, 99)) :: v4
end module m1

module m2
  type baseType(baseParam)
    integer, kind :: baseParam = 4
  end type baseType
  type dtype(dtypeParam)
    integer, kind :: dtypeParam = 4
    type(baseType(dtypeParam)) :: baseField
    !ERROR: KIND parameter value (343) of intrinsic type REAL did not resolve to a supported value
    real(baseField%baseParam) :: realField
  end type dtype

  type(dtype) :: v1
  type(dtype(8)) :: v2
  type(dtype(343)) :: v3
end module m2

module m3
  type dtype(goodDefaultLen, badDefaultLen)
    integer, len :: goodDefaultLen = 4
    integer, len :: badDefaultLen = 343
  end type dtype
  type(dtype) :: v1
  type(dtype(4, 4)) :: v2
  type(dtype(99, 4)) :: v3
  type(dtype(4, 99)) :: v4
  real(v1%goodDefaultLen), pointer :: pGood1
  !ERROR: REAL(KIND=343) is not a supported type
  real(v1%badDefaultLen), pointer :: pBad1
  real(v2%goodDefaultLen), pointer :: pGood2
  real(v2%badDefaultLen), pointer :: pBad2
  !ERROR: REAL(KIND=99) is not a supported type
  real(v3%goodDefaultLen), pointer :: pGood3
  real(v3%badDefaultLen), pointer :: pBad3
  real(v4%goodDefaultLen), pointer :: pGood4
  !ERROR: REAL(KIND=99) is not a supported type
  real(v4%badDefaultLen), pointer :: pBad4
end module m3
