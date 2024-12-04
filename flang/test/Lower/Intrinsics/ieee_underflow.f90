! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

! CHECK-LABEL: c.func @_QPs
subroutine s
  ! CHECK:     %[[V_0:[0-9]+]] = fir.call @fetestexcept(%c-1{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:     %[[V_1:[0-9]+]] = fir.call @feclearexcept(%[[V_0]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     %[[V_2:[0-9]+]] = fir.call @_FortranAGetUnderflowMode() fastmath<contract> : () -> i1
  use ieee_arithmetic, only: ieee_get_underflow_mode, ieee_set_underflow_mode

  ! CHECK:     %[[V_3:[0-9]+]] = fir.alloca !fir.logical<4> {bindc_name = "r", uniq_name = "_QFsEr"}
  ! CHECK:     %[[V_4:[0-9]+]]:2 = hlfir.declare %[[V_3]] {uniq_name = "_QFsEr"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
  logical r

  ! CHECK:     %[[V_5:[0-9]+]] = fir.convert %false{{[_0-9]*}} : (i1) -> i1
  ! CHECK:     %[[V_6:[0-9]+]] = fir.call @_FortranASetUnderflowMode(%[[V_5]]) fastmath<contract> : (i1) -> none
  call ieee_set_underflow_mode(.false.)

  ! CHECK:     %[[V_7:[0-9]+]] = fir.call @_FortranAGetUnderflowMode() fastmath<contract> : () -> i1
  ! CHECK:     %[[V_8:[0-9]+]] = fir.convert %[[V_7]] : (i1) -> !fir.logical<4>
  ! CHECK:     fir.store %[[V_8]] to %[[V_4]]#1 : !fir.ref<!fir.logical<4>>
  call ieee_get_underflow_mode(r)
! print*, r

  ! CHECK:     %[[V_9:[0-9]+]] = fir.convert %true{{[_0-9]*}} : (i1) -> i1
  ! CHECK:     %[[V_10:[0-9]+]] = fir.call @_FortranASetUnderflowMode(%[[V_9]]) fastmath<contract> : (i1) -> none
  call ieee_set_underflow_mode(.true.)

  ! CHECK:     %[[V_11:[0-9]+]] = fir.call @_FortranAGetUnderflowMode() fastmath<contract> : () -> i1
  ! CHECK:     %[[V_12:[0-9]+]] = fir.convert %[[V_11]] : (i1) -> !fir.logical<4>
  ! CHECK:     fir.store %[[V_12]] to %[[V_4]]#1 : !fir.ref<!fir.logical<4>>
  call ieee_get_underflow_mode(r)
! print*, r

  ! CHECK:     %[[V_13:[0-9]+]] = fir.call @_FortranASetUnderflowMode(%[[V_2]]) fastmath<contract> : (i1) -> none
  ! CHECK:     %[[V_14:[0-9]+]] = fir.call @feraiseexcept(%[[V_0]]) fastmath<contract> : (i32) -> i32
end

  call s
end
