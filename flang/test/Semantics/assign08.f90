! RUN: %python %S/test_errors.py %s %flang_fc1
! "Same type" checking for intrinsic assignment
module m1
  type :: nonSeqType
    integer :: n1
  end type
  type :: seqType
    sequence
    integer :: n2
  end type
  type, bind(c) :: bindCType
    integer :: n3
  end type
end module

program test
  use m1, modNonSeqType => nonSeqType, modSeqType => seqType, modBindCType => bindCType
  type :: nonSeqType
    integer :: n1
  end type
  type :: seqType
    sequence
    integer :: n2
  end type
  type, bind(c) :: bindCType
    integer :: n3
  end type
  type(modNonSeqType) :: mns1, mns2
  type(modSeqType) :: ms1, ms2
  type(modBindCType) :: mb1, mb2
  type(nonSeqType) :: ns1, ns2
  type(seqType) :: s1, s2
  type(bindCType) :: b1, b2
  ! These are trivially ok
  mns1 = mns2
  ms1 = ms2
  mb1 = mb2
  ns1 = ns2
  s1 = s2
  b1 = b2
  ! These are ok per 7.5.2.4
  ms1 = s1
  mb1 = b1
  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types TYPE(modnonseqtype) and TYPE(nonseqtype)
  mns1 = ns1
  !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types TYPE(nonseqtype) and TYPE(modnonseqtype)
  ns1 = mns1
end
