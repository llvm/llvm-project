! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=COARRAY
! RUN: not %flang_fc1 -emit-hlfir %s 2>&1 | FileCheck %s --check-prefixes=NOCOARRAY

program test_team_number
  use, intrinsic :: iso_fortran_env, only: team_type
  implicit none
  ! NOCOARRAY: Not yet implemented: Multi-image features are experimental and are disabled by default, use '-fcoarray' to enable.

  type(team_type) :: team
  integer :: t
 
  ! COARRAY: %[[RES:.*]] = mif.team_number team %[[TEAM:.*]] : ({{.*}}) -> i64
  t = team_number(team)
  
  ! COARRAY: %[[RES:.*]] = mif.team_number : () -> i64
  t = team_number()
  
end program test_team_number
 
