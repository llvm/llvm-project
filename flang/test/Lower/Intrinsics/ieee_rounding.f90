! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

! CHECK-LABEL: func.func @_QQmain
program r
  use ieee_arithmetic
  ! CHECK:     %[[V_56:[0-9]+]] = fir.alloca !fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}> {bindc_name = "round_value", uniq_name = "_QFEround_value"}
  ! CHECK:     %[[V_57:[0-9]+]]:2 = hlfir.declare %[[V_56]] {uniq_name = "_QFEround_value"}
  type(ieee_round_type) :: round_value

  ! CHECK:     fir.if %true{{[_0-9]*}} {
  if (ieee_support_rounding(ieee_down)) then
    ! CHECK:       %[[V_62:[0-9]+]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_round_type.mode, !fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>
    ! CHECK:       %[[V_63:[0-9]+]] = fir.coordinate_of %[[V_57]]#1, %[[V_62]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
    ! CHECK:       %[[V_64:[0-9]+]] = fir.call @llvm.get.rounding() fastmath<contract> : () -> i32
    ! CHECK:       %[[V_65:[0-9]+]] = fir.convert %[[V_64]] : (i32) -> i8
    ! CHECK:       fir.store %[[V_65]] to %[[V_63]] : !fir.ref<i8>
    call ieee_get_rounding_mode(round_value)

    ! CHECK:       %[[V_66:[0-9]+]] = fir.address_of(@_QQro._QM__fortran_builtinsT__builtin_ieee_round_type.0) : !fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>
    ! CHECK:       %[[V_67:[0-9]+]]:2 = hlfir.declare %[[V_66]]
    ! CHECK:       %[[V_68:[0-9]+]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_round_type.mode, !fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>
    ! CHECK:       %[[V_69:[0-9]+]] = fir.coordinate_of %[[V_67]]#1, %[[V_68]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
    ! CHECK:       %[[V_70:[0-9]+]] = fir.load %[[V_69]] : !fir.ref<i8>
    ! CHECK:       %[[V_71:[0-9]+]] = arith.shli %c-1{{.*}}, %c2{{.*}} : i8
    ! CHECK:       %[[V_72:[0-9]+]] = arith.andi %[[V_70]], %[[V_71]] : i8
    ! CHECK:       %[[V_73:[0-9]+]] = arith.cmpi eq, %[[V_72]], %c0{{.*}} : i8
    ! CHECK:       %[[V_74:[0-9]+]] = arith.select %[[V_73]], %[[V_70]], %c1{{.*}} : i8
    ! CHECK:       %[[V_75:[0-9]+]] = fir.convert %[[V_74]] : (i8) -> i32
    ! CHECK:       fir.call @llvm.set.rounding(%[[V_75]]) fastmath<contract> : (i32) -> ()
    call ieee_set_rounding_mode(ieee_down)
    print*, 'ok'

    ! CHECK:       %[[V_85:[0-9]+]] = fir.field_index _QM__fortran_builtinsT__builtin_ieee_round_type.mode, !fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>
    ! CHECK:       %[[V_86:[0-9]+]] = fir.coordinate_of %[[V_57]]#1, %[[V_85]] : (!fir.ref<!fir.type<_QM__fortran_builtinsT__builtin_ieee_round_type{_QM__fortran_builtinsT__builtin_ieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
    ! CHECK:       %[[V_87:[0-9]+]] = fir.load %[[V_86]] : !fir.ref<i8>
    ! CHECK:       %[[V_88:[0-9]+]] = arith.shli %c-1{{.*}}, %c2{{.*}} : i8
    ! CHECK:       %[[V_89:[0-9]+]] = arith.andi %[[V_87]], %[[V_88]] : i8
    ! CHECK:       %[[V_90:[0-9]+]] = arith.cmpi eq, %[[V_89]], %c0{{.*}} : i8
    ! CHECK:       %[[V_91:[0-9]+]] = arith.select %[[V_90]], %[[V_87]], %c1{{.*}} : i8
    ! CHECK:       %[[V_92:[0-9]+]] = fir.convert %[[V_91]] : (i8) -> i32
    ! CHECK:       fir.call @llvm.set.rounding(%[[V_92]]) fastmath<contract> : (i32) -> ()
    call ieee_set_rounding_mode(round_value)
  endif
end
