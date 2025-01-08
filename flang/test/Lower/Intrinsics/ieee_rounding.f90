! RUN: bbc -emit-fir -hlfir=false -o - %s | FileCheck %s

! CHECK-LABEL: c.func @_QQmain
program r
  use ieee_arithmetic
  ! CHECK-DAG: %[[V_0:[0-9]+]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>
  ! CHECK-DAG: %[[V_2:[0-9]+]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}> {bindc_name = "round_value", uniq_name = "_QFEround_value"}
  type(ieee_round_type) :: round_value

  ! CHECK:     fir.if %true {
  if (ieee_support_rounding(ieee_down)) then
    ! CHECK:       %[[V_23:[0-9]+]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_round_type.mode, !fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>
    ! CHECK:       %[[V_24:[0-9]+]] = fir.coordinate_of %[[V_2]], %[[V_23]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
    ! CHECK:       %[[V_25:[0-9]+]] = fir.call @llvm.get.rounding() {{.*}} : () -> i32
    ! CHECK:       %[[V_26:[0-9]+]] = fir.convert %[[V_25]] : (i32) -> i8
    ! CHECK:       fir.store %[[V_26]] to %[[V_24]] : !fir.ref<i8>
    call ieee_get_rounding_mode(round_value)

    ! CHECK:       %[[V_32:[0-9]+]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_round_type.mode, !fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>
    ! CHECK:       %[[V_33:[0-9]+]] = fir.coordinate_of %[[V_0]], %[[V_32]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
    ! CHECK:       fir.store %c3{{.*}} to %[[V_33]] : !fir.ref<i8>
    ! CHECK:       %[[V_34:[0-9]+]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_round_type.mode, !fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>
    ! CHECK:       %[[V_35:[0-9]+]] = fir.coordinate_of %[[V_0]], %[[V_34]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
    ! CHECK:       %[[V_36:[0-9]+]] = fir.load %[[V_35]] : !fir.ref<i8>
    ! CHECK:       %[[V_37:[0-9]+]] = fir.convert %[[V_36]] : (i8) -> i32
    ! CHECK:       fir.call @llvm.set.rounding(%[[V_37]]) {{.*}} : (i32) -> ()
    call ieee_set_rounding_mode(ieee_down)
    print*, 'ok'

    ! CHECK:       %[[V_46:[0-9]+]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_round_type.mode, !fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>
    ! CHECK:       %[[V_47:[0-9]+]] = fir.coordinate_of %[[V_2]], %[[V_46]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
    ! CHECK:       %[[V_48:[0-9]+]] = fir.load %[[V_47]] : !fir.ref<i8>
    ! CHECK:       %[[V_49:[0-9]+]] = fir.convert %[[V_48]] : (i8) -> i32
    ! CHECK:       fir.call @llvm.set.rounding(%[[V_49]]) {{.*}} : (i32) -> ()
    call ieee_set_rounding_mode(round_value)
  endif
end
