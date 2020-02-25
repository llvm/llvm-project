! Error test -- DO loop uses obsolete loop termination statement
! See R1131 and C1133

! By default, this is not an error and label do are rewritten to non-label do.
! A warning is generated with -Mstandard

! RUN: ${F18} -funparse-with-symbols -Mstandard -I../../tools/f18/include %s 2>&1 | ${FileCheck} %s

! CHECK: end do

! The following CHECK-NOT actively uses the fact that the leading zero of labels
! would be removed in the unparse but not the line linked to warnings. We do
! not want to see label do in the unparse only.
! CHECK-NOT: do [1-9]

! CHECK: A DO loop should terminate with an END DO or CONTINUE

subroutine foo8()
  use iso_fortran_env, only : team_type
  type(team_type) :: odd_even
  do 01 k=1,10
    change team (odd_even)
01  end team
end subroutine
