! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=COARRAY
! RUN: not %flang_fc1 -emit-hlfir %s 2>&1 | FileCheck %s --check-prefixes=NOCOARRAY

program test_get_team
  use, intrinsic :: iso_fortran_env, only: team_type, initial_team, current_team, parent_team
  implicit none
  ! NOCOARRAY: Not yet implemented: Multi-image features are experimental and are disabled by default, use '-fcoarray' to enable.

  type(team_type) :: result_team
  integer :: n 

  ! COARRAY: %[[RES:.*]] = mif.get_team : () -> {{.*}}
  result_team = get_team()

  ! COARRAY: %[[RES:.*]] = mif.get_team level %[[INIT:.*]] : (i32) -> {{.*}}
  result_team = get_team(initial_team)

  ! COARRAY: %[[RES:.*]] = mif.get_team level %[[CURRENT:.*]] : (i32) -> {{.*}}
  result_team = get_team(current_team)

  ! COARRAY: %[[RES:.*]] = mif.get_team level %[[PARENT:.*]] : (i32) -> {{.*}}
  result_team = get_team(parent_team)

  ! COARRAY: %[[RES:.*]] = mif.get_team level %[[VAL_N:.*]] : (i32) -> {{.*}}
  result_team = get_team(n) 

end program test_get_team
 
