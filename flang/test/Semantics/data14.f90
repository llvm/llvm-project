! RUN: %flang_fc1 -fsyntax-only %s 2>&1 | FileCheck %s
! Verify varnings on nonconforming DATA statements
! As a common extension, C876 violations are not errors.
program main
  type :: seqType
    sequence
    integer :: number
  end type
  type(seqType) :: x
  integer :: j
  common j, x, y
  !CHECK: Blank COMMON object 'j' in a DATA statement is not standard
  data j/1/
  !CHECK: Blank COMMON object 'x' in a DATA statement is not standard
  data x%number/2/
end
