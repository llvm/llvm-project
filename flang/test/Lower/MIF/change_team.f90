! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=COARRAY
! RUN: not %flang_fc1 -emit-hlfir %s 2>&1 | FileCheck %s --check-prefixes=NOCOARRAY

program test_change_team
  use, intrinsic :: iso_fortran_env, only: team_type
  implicit none
  ! NOCOARRAY: Not yet implemented: Multi-image features are experimental and are disabled by default, use '-fcoarray' to enable.

  type(team_type) :: team
  integer :: stat, i
  character(len=10) :: err

  ! COARRAY: mif.change_team %[[TEAM:.*]] : ({{.*}}) {
  change team (team)
    i = i +1
  end team 
  ! COARRAY: mif.end_team 
  ! COARRAY: }
 
  ! COARRAY: mif.change_team %[[TEAM:.*]] stat %[[STAT:.*]]#0 errmsg %[[ERR:.*]] : ({{.*}}, !fir.ref<i32>, !fir.box<!fir.char<1,10>>) {
  change team (team, STAT=stat, ERRMSG=err)
  end team
  ! COARRAY: mif.end_team 
  ! COARRAY: }

end program test_change_team
 
