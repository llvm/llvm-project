! RUN: %python %S/test_errors.py %s %flang_fc1
! C929   No specifier shall appear more than once in a given 
!   image-selector-spec-list.
! C930 TEAM and TEAM_NUMBER shall not both appear in the same
!   image-selector-spec-list.
! C931 A stat-variable in an image-selector shall not be a coindexed object.
subroutine s1()
  use ISO_FORTRAN_ENV
  type(team_type) :: team1, team2
  real :: rCoarray[10,20,*]
  real :: rVar1, rVar2
  integer :: iVar1, iVar2
  integer, dimension(4) :: intArray
  integer :: intScalarCoarray[*]
  integer :: intCoarray[3, 4, *]
  integer :: smallIntCoarray[4, *]
  intCoVar = 343
  ! OK
  rVar1 = rCoarray[1,2,3]
  !ERROR: 'rcoarray' has corank 3, but coindexed reference has 2 cosubscripts
  rVar1 = rCoarray[1,2]
  !ERROR: Must have INTEGER type, but is REAL(4)
  rVar1 = rCoarray[1,2,3.4]
  !ERROR: Must have INTEGER type, but is REAL(4)
  iVar1 = smallIntCoarray[3.4]
  !ERROR: Must be a scalar value, but is a rank-1 array
  rVar1 = rCoarray[1,intArray,3]
  ! OK
  rVar1 = rCoarray[1,2,3,STAT=iVar1, TEAM=team2]
  !ERROR: Team value must be of type TEAM_TYPE from module ISO_FORTRAN_ENV
  rVar1 = rCoarray[1,2,3,STAT=iVar1, TEAM=2]
  ! OK
  rVar1 = rCoarray[1,2,3,STAT=iVar1, TEAM_NUMBER=38]
  ! OK
  rVar1 = rCoarray[1,2,3,STAT=iVar1]
  ! OK
  rVar1 = rCoarray[1,2,3,STAT=intArray(2)]
  !ERROR: Must have INTEGER type, but is REAL(4)
  rVar1 = rCoarray[1,2,3,STAT=rVar2]
  !ERROR: Must be a scalar value, but is a rank-1 array
  rVar1 = rCoarray[1,2,3,STAT=intArray]
  ! Error on C929, no specifier can appear more than once
  !ERROR: STAT variable can only be specified once
  rVar1 = rCoarray[1,2,3,STAT=iVar1, STAT=iVar2]
  ! OK
  rVar1 = rCoarray[1,2,3,TEAM=team1]
  ! Error on C929, no specifier can appear more than once
  !ERROR: TEAM value can only be specified once
  rVar1 = rCoarray[1,2,3,TEAM=team1, TEAM=team2]
  ! OK
  rVar1 = rCoarray[1,2,3,TEAM_NUMBER=37]
  ! OK
  rVar1 = rCoarray[1,2,3,TEAM_NUMBER=iVar1]
  ! Error, team number is a scalar integer expression
  !ERROR: Must be a scalar value, but is a rank-1 array
  rVar1 = rCoarray[1,2,3,TEAM_NUMBER=intArray]
  ! Error, team number is a scalar integer expression
  !ERROR: Must have INTEGER type, but is REAL(4)
  rVar1 = rCoarray[1,2,3,TEAM_NUMBER=3.7]
  ! Error on C929, no specifier can appear more than once
  !ERROR: TEAM_NUMBER value can only be specified once
  rVar1 = rCoarray[1,2,3,TEAM_NUMBER=37, TEAM_NUMBER=37]
  !ERROR: Cannot specify both TEAM and TEAM_NUMBER
  rVar1 = rCoarray[1,2,3,TEAM=team1, TEAM_NUMBER=37]
  !ERROR: Cannot specify both TEAM and TEAM_NUMBER
  rVar1 = rCoarray[1,2,3,TEAM_number=43, TEAM=team1]
  ! OK for a STAT variable to be a coarray integer
  rVar1 = rCoarray[1,2,3,stat=intScalarCoarray]
  ! Error for a STAT variable to be a coindexed object
  !ERROR: Image selector STAT variable must not be a coindexed object
  rVar1 = rCoarray[1,2,3,stat=intCoarray[2,3, 4]]
end subroutine s1
