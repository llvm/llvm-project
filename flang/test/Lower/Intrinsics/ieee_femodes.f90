! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: c.func @_QQmain
program m
  use ieee_arithmetic
  use ieee_exceptions

  ! CHECK:  %[[VAL_69:.*]] = fir.alloca !fir.type<_QM__fortran_ieee_exceptionsTieee_modes_type{_QM__fortran_ieee_exceptionsTieee_modes_type.__data:!fir.array<2xi32>}> {bindc_name = "modes", uniq_name = "_QFEmodes"}
  ! CHECK:  %[[VAL_70:.*]] = fir.declare %[[VAL_69]] {uniq_name = "_QFEmodes"} : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_modes_type{_QM__fortran_ieee_exceptionsTieee_modes_type.__data:!fir.array<2xi32>}>>) -> !fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_modes_type{_QM__fortran_ieee_exceptionsTieee_modes_type.__data:!fir.array<2xi32>}>>
  type(ieee_modes_type) :: modes

  ! CHECK:  %[[VAL_71:.*]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}> {bindc_name = "round", uniq_name = "_QFEround"}
  ! CHECK:  %[[VAL_72:.*]] = fir.declare %[[VAL_71]] {uniq_name = "_QFEround"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>
  type(ieee_round_type) :: round

  ! CHECK:  %[[VAL_78:.*]] = fir.address_of(@_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.0) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>
  ! CHECK:  %[[VAL_79:.*]] = fir.declare %[[VAL_78]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.0"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>

  ! CHECK:  %[[VAL_80:.*]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_round_type.mode, !fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>
  ! CHECK:  %[[VAL_81:.*]] = fir.coordinate_of %[[VAL_79]], %[[VAL_80]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:  %[[VAL_82:.*]] = fir.load %[[VAL_81]] : !fir.ref<i8>
  ! CHECK:  %[[VAL_83:.*]] = fir.convert %[[VAL_82]] : (i8) -> i32
  ! CHECK:  fir.call @llvm.set.rounding(%[[VAL_83]]) fastmath<contract> : (i32) -> ()
  call ieee_set_rounding_mode(ieee_up)

  ! CHECK:  %[[VAL_84:.*]] = fir.coordinate_of %[[VAL_72]], %[[VAL_80]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:  %[[VAL_85:.*]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:  %[[VAL_86:.*]] = fir.convert %[[VAL_85]] : (i32) -> i8
  ! CHECK:  fir.store %[[VAL_86]] to %[[VAL_84]] : !fir.ref<i8>
  call ieee_get_rounding_mode(round)

  print*, 'rounding_mode [up     ] : ', mode_name(round)

  ! CHECK:  %[[VAL_103:.*]] = fir.convert %[[VAL_70]] : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_modes_type{_QM__fortran_ieee_exceptionsTieee_modes_type.__data:!fir.array<2xi32>}>>) -> !fir.ref<i32>
  ! CHECK:  %[[VAL_104:.*]] = fir.call @fegetmode(%[[VAL_103]]) fastmath<contract> : (!fir.ref<i32>) -> i32
  call ieee_get_modes(modes)

  ! CHECK:  %[[VAL_105:.*]] = fir.address_of(@_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.1) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>
  ! CHECK:  %[[VAL_106:.*]] = fir.declare %[[VAL_105]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.1"} : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>) -> !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>
  ! CHECK:  %[[VAL_107:.*]] = fir.coordinate_of %[[VAL_106]], %[[VAL_80]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:  %[[VAL_108:.*]] = fir.load %[[VAL_107]] : !fir.ref<i8>
  ! CHECK:  %[[VAL_109:.*]] = fir.convert %[[VAL_108]] : (i8) -> i32
  ! CHECK:  fir.call @llvm.set.rounding(%[[VAL_109]]) fastmath<contract> : (i32) -> ()
  call ieee_set_rounding_mode(ieee_to_zero)

  ! CHECK:  %[[VAL_110:.*]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:  %[[VAL_111:.*]] = fir.convert %[[VAL_110]] : (i32) -> i8
  ! CHECK:  fir.store %[[VAL_111]] to %[[VAL_84]] : !fir.ref<i8>
  call ieee_get_rounding_mode(round)

  print*, 'rounding_mode [to_zero] : ', mode_name(round)

  ! CHECK:  %[[VAL_126:.*]] = fir.call @fesetmode(%[[VAL_103]]) fastmath<contract> : (!fir.ref<i32>) -> i32
  call ieee_set_modes(modes)

  ! CHECK:  %[[VAL_127:.*]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:  %[[VAL_128:.*]] = fir.convert %[[VAL_127]] : (i32) -> i8
  ! CHECK:  fir.store %[[VAL_128]] to %[[VAL_84]] : !fir.ref<i8>
  call ieee_get_rounding_mode(round)

  print*, 'rounding_mode [up     ] : ', mode_name(round)

contains
  character(7) function mode_name(m)
    type(ieee_round_type), intent(in) :: m
    if (m == ieee_nearest) then
      mode_name = 'nearest'
    else if (m == ieee_to_zero) then
      mode_name = 'to_zero'
    else if (m == ieee_up) then
      mode_name = 'up'
    else if (m == ieee_down) then
      mode_name = 'down'
    else if (m == ieee_away) then
      mode_name = 'away'
    else if (m == ieee_other) then
      mode_name = 'other'
    else
      mode_name = '???'
    endif
  end
end
