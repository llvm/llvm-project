! RUN: %S/test_errors.sh %s %t %f18
! Test BLOCK DATA subprogram (14.3)
block data foo
  !ERROR: IMPORT is not allowed in a BLOCK DATA subprogram
  import
  real :: pi = asin(-1.0) ! ok
  !ERROR: An initialized variable in BLOCK DATA must be in a COMMON block
  integer :: notInCommon = 1
  integer :: uninitialized ! ok
  !ERROR: 'p' may not appear in a BLOCK DATA subprogram
  procedure(sin), pointer :: p => cos
  !ERROR: 'p' is already declared as a procedure
  common /block/ pi, p
  !ERROR: An initialized variable in BLOCK DATA must be in a COMMON block
  integer :: inDataButNotCommon
  data inDataButNotCommon /1/
  !ERROR: Two objects in the same EQUIVALENCE set may not be members of distinct COMMON blocks
  integer :: inCommonA, inCommonB
  common /a/ inCommonA, /b/ inCommonB
  equivalence(inCommonA, inCommonB)
  integer :: inCommonD, initialized ! ok
  common /d/ inCommonD
  equivalence(inCommonD, initialized)
  data initialized /2/
end block data
