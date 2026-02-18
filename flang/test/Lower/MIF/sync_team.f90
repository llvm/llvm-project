! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=COARRAY
! RUN2: not %flang_fc1 -emit-hlfir %s -o - | FileCheck %s --check-prefixes=NOCOARRAY

program test_sync_team
  use, intrinsic :: iso_fortran_env, only: team_type
  implicit none
  ! NOCOARRAY: Not yet implemented: Multi-image features are experimental and are disabled by default, use '-fcoarray' to enable.
 
  integer sync_status
  character(len=128) :: error_message
  type(team_type) :: team

  ! COARRAY: mif.sync_team %[[TEAM:.*]] : ({{.*}}) -> ()
  sync team(team)

  ! COARRAY: mif.sync_team %[[TEAM:.*]] stat %[[STAT:.*]]#0 : ({{.*}}, !fir.ref<i32>) -> ()
  sync team(team, stat=sync_status)
  
  ! COARRAY: mif.sync_team %[[TEAM:.*]] errmsg %[[ERR:.*]] : ({{.*}}, !fir.box<!fir.char<1,128>>) -> ()
  sync team(team,                    errmsg=error_message)
  
  ! COARRAY: mif.sync_team %[[TEAM:.*]] stat %[[STAT:.*]]#0 errmsg %[[ERR:.*]] : ({{.*}}, !fir.ref<i32>, !fir.box<!fir.char<1,128>>) -> ()
  sync team(team, stat=sync_status, errmsg=error_message)

end program test_sync_team
