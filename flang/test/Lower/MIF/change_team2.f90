! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s --check-prefixes=COARRAY
! RUN: not %flang_fc1 -emit-hlfir %s 2>&1 | FileCheck %s --check-prefixes=NOCOARRAY

! NOCOARRAY: Not yet implemented: Multi-image features are experimental and are disabled by default, use '-fcoarray' to enable.

  use iso_fortran_env, only : team_type, STAT_FAILED_IMAGE
  implicit none
  type(team_type) :: team
  integer         :: new_team, image_status
  new_team = mod(this_image(),2)+1
  form team (new_team,team)
  change team (team)
    if (team_number() /= new_team) STOP 1
  end team
  ! COARRAY:  mif.change_team %[[TEAM:.*]]  : ({{.*}}) {
  ! COARRAY:     %[[VAL_1:.*]] = mif.team_number : () -> i64
  ! COARRAY:     %[[VAL_2:.*]] = fir.convert %[[VAL_1]] : (i64) -> i32
  ! COARRAY:     %[[VAL_3:.*]] = fir.load %[[VAR_1:.*]]#0 : !fir.ref<i32>
  ! COARRAY:     %[[VAL_4:.*]] = arith.cmpi ne, %[[VAL_2]], %[[VAL_3]] : i32
  ! COARRAY:     cf.cond_br %[[VAL_4]], ^bb1, ^bb2
  ! COARRAY:   ^bb1:  // pred: ^bb0
  ! COARRAY:     %[[C1_I32:.*]] = arith.constant 1 : i32
  ! COARRAY:     %[[FALSE_1:.*]] = arith.constant false
  ! COARRAY:     %[[FALSE_2:.*]] = arith.constant false
  ! COARRAY:     fir.call @_FortranAStopStatement(%[[C1_I32]], %[[FALSE_1]], %[[FALSE_2]]) fastmath<contract> : (i32, i1, i1) -> ()
  ! COARRAY:     fir.unreachable
  ! COARRAY:   ^bb2:  // pred: ^bb0
  ! COARRAY:     mif.end_team : () -> ()
  ! COARRAY:   }

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
