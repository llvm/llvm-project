! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=COARRAY
! RUN: not %flang_fc1 -emit-hlfir %s 2>&1 | FileCheck %s --check-prefixes=NOCOARRAY

program test_form_team
  use, intrinsic :: iso_fortran_env, only: team_type
  implicit none
  ! NOCOARRAY: Not yet implemented: Multi-image features are experimental and are disabled by default, use '-fcoarray' to enable.

  type(team_type) :: team
  integer :: team_number
  integer :: team_index
  integer :: stat
  character(len=10) :: err

  form team (team_number, team)
  ! COARRAY: mif.form_team team_number %[[ARG1:.*]] team_var %[[ARG2:.*]] : (i32, {{.*}}) -> ()
  
  form team (team_number, team, NEW_INDEX=team_index)
  ! COARRAY: mif.form_team team_number %[[ARG1:.*]] team_var %[[ARG2:.*]] new_index %[[NI:.*]] : (i32, {{.*}}, i32) -> ()
  
  form team (team_number, team, STAT=stat)
  ! COARRAY: mif.form_team team_number %[[ARG1:.*]] team_var %[[ARG2:.*]] stat %[[STAT:.*]] : (i32, {{.*}}, !fir.ref<i32>) -> ()
  
  form team (team_number, team, ERRMSG=err)
  ! COARRAY: mif.form_team team_number %[[ARG1:.*]] team_var %[[ARG2:.*]] errmsg %[[ERR:.*]] : (i32, {{.*}}, !fir.box<!fir.char<1,10>>) -> ()

end program test_form_team

 
