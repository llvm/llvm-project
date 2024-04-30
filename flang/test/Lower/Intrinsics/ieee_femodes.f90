! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: c.func @_QQmain
program m
  use ieee_arithmetic
  use ieee_exceptions

  ! CHECK:  %[[V_59:[0-9]+]] = fir.alloca !fir.type<_QM__fortran_ieee_exceptionsTieee_modes_type{_QM__fortran_ieee_exceptionsTieee_modes_type.__data:!fir.array<2xi32>}> {bindc_name = "modes", uniq_name = "_QFEmodes"}
  ! CHECK:  %[[V_60:[0-9]+]] = fir.declare %[[V_59]] {uniq_name = "_QFEmodes"} : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_modes_type{_QM__fortran_ieee_exceptionsTieee_modes_type.__data:!fir.array<2xi32>}>>) -> !fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_modes_type{_QM__fortran_ieee_exceptionsTieee_modes_type.__data:!fir.array<2xi32>}>>
  type(ieee_modes_type) :: modes

  ! CHECK:  %[[V_61:[0-9]+]] = fir.alloca !fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}> {bindc_name = "round", uniq_name = "_QFEround"}
  ! CHECK:  %[[V_62:[0-9]+]] = fir.declare %[[V_61]] {uniq_name = "_QFEround"} : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>) -> !fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>
  type(ieee_round_type) :: round

  ! CHECK:  %[[V_68:[0-9]+]] = fir.address_of(@_QQro._QMieee_arithmeticTieee_round_type.0) : !fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>
  ! CHECK:  %[[V_69:[0-9]+]] = fir.declare %[[V_68]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QMieee_arithmeticTieee_round_type.0"} : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>) -> !fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>

  ! CHECK:  %[[V_70:[0-9]+]] = fir.field_index _QMieee_arithmeticTieee_round_type.mode, !fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>
  ! CHECK:  %[[V_71:[0-9]+]] = fir.coordinate_of %[[V_69]], %[[V_70]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:  %[[V_72:[0-9]+]] = fir.load %[[V_71]] : !fir.ref<i8>
  ! CHECK:  %[[V_73:[0-9]+]] = fir.convert %[[V_72]] : (i8) -> i32
  ! CHECK:  fir.call @llvm.set.rounding(%[[V_73]]) fastmath<contract> : (i32) -> ()
  call ieee_set_rounding_mode(ieee_up)

  ! CHECK:  %[[V_74:[0-9]+]] = fir.coordinate_of %[[V_62]], %[[V_70]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:  %[[V_75:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:  %[[V_76:[0-9]+]] = fir.convert %[[V_75]] : (i32) -> i8
  ! CHECK:  fir.store %[[V_76]] to %[[V_74]] : !fir.ref<i8>
  call ieee_get_rounding_mode(round)

  print*, 'rounding_mode [up     ] : ', mode_name(round)

  ! CHECK:  %[[V_93:[0-9]+]] = fir.convert %[[V_60]] : (!fir.ref<!fir.type<_QM__fortran_ieee_exceptionsTieee_modes_type{_QM__fortran_ieee_exceptionsTieee_modes_type.__data:!fir.array<2xi32>}>>) -> !fir.ref<i32>
  ! CHECK:  %[[V_94:[0-9]+]] = fir.call @fegetmode(%[[V_93]]) fastmath<contract> : (!fir.ref<i32>) -> i32
  call ieee_get_modes(modes)

  ! CHECK:  %[[V_95:[0-9]+]] = fir.address_of(@_QQro._QMieee_arithmeticTieee_round_type.1) : !fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>
  ! CHECK:  %[[V_96:[0-9]+]] = fir.declare %[[V_95]] {fortran_attrs = #fir.var_attrs<parameter>, uniq_name = "_QQro._QMieee_arithmeticTieee_round_type.1"} : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>) -> !fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>
  ! CHECK:  %[[V_97:[0-9]+]] = fir.coordinate_of %[[V_96]], %[[V_70]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:  %[[V_98:[0-9]+]] = fir.load %[[V_97]] : !fir.ref<i8>
  ! CHECK:  %[[V_99:[0-9]+]] = fir.convert %[[V_98]] : (i8) -> i32
  ! CHECK:  fir.call @llvm.set.rounding(%[[V_99]]) fastmath<contract> : (i32) -> ()
  call ieee_set_rounding_mode(ieee_to_zero)

  ! CHECK:  %[[V_100:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:  %[[V_101:[0-9]+]] = fir.convert %[[V_100]] : (i32) -> i8
  ! CHECK:  fir.store %[[V_101]] to %[[V_74]] : !fir.ref<i8>
  call ieee_get_rounding_mode(round)

  print*, 'rounding_mode [to_zero] : ', mode_name(round)

  ! CHECK:  %[[V_116:[0-9]+]] = fir.call @fesetmode(%[[V_93]]) fastmath<contract> : (!fir.ref<i32>) -> i32
  call ieee_set_modes(modes)

  ! CHECK:  %[[V_117:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
  ! CHECK:  %[[V_118:[0-9]+]] = fir.convert %[[V_117]] : (i32) -> i8
  ! CHECK:  fir.store %[[V_118]] to %[[V_74]] : !fir.ref<i8>
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
