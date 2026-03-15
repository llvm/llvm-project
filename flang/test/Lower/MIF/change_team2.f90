! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=COARRAY
! RUN: not %flang_fc1 -emit-hlfir %s 2>&1 | FileCheck %s --check-prefixes=NOCOARRAY

! NOCOARRAY: Not yet implemented: Multi-image features are experimental and are disabled by default, use '-fcoarray' to enable.

  use iso_fortran_env, only : team_type, STAT_FAILED_IMAGE
  implicit none
  type(team_type) :: team
  integer         :: new_team, image_status
  new_team = mod(this_image(),2)+1
  form team (new_team,team)
  ! COARRAY: mif.change_team %[[TEAM:.*]] : ({{.*}}) {
  change team (team)
    if (team_number() /= new_team) STOP 1
  end team
  ! COARRAY: mif.end_team 
  ! COARRAY: }
  if (runtime_popcnt(0_16) /= 0) STOP 2
  if (runtime_poppar(1_16) /= 1) STOP 3
contains
  integer function runtime_popcnt (i)
    integer(kind=16), intent(in) :: i
    runtime_popcnt = popcnt(i)
  end function
  integer function runtime_poppar (i)
    integer(kind=16), intent(in) :: i
    runtime_poppar = poppar(i)
  end function
end
