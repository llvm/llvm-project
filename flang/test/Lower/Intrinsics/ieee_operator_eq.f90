! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: c.func @_QPs
subroutine s(r1,r2)
  use ieee_arithmetic, only: ieee_round_type, operator(==)
  type(ieee_round_type) :: r1, r2
  ! CHECK:   %[[V_3:[0-9]+]] = fir.field_index _QMieee_arithmeticTieee_round_type.mode, !fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>
  ! CHECK:   %[[V_4:[0-9]+]] = fir.coordinate_of %arg0, %[[V_3]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:   %[[V_5:[0-9]+]] = fir.field_index _QMieee_arithmeticTieee_round_type.mode, !fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>
  ! CHECK:   %[[V_6:[0-9]+]] = fir.coordinate_of %arg1, %[[V_5]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:   %[[V_7:[0-9]+]] = fir.load %[[V_4]] : !fir.ref<i8>
  ! CHECK:   %[[V_8:[0-9]+]] = fir.load %[[V_6]] : !fir.ref<i8>
  ! CHECK:   %[[V_9:[0-9]+]] = arith.cmpi eq, %[[V_7]], %[[V_8]] : i8
  ! CHECK:   %[[V_10:[0-9]+]] = fir.call @_FortranAioOutputLogical(%{{.*}} %[[V_9]]) {{.*}} : (!fir.ref<i8>, i1) -> i1
  ! CHECK:   return
  ! CHECK: }
  print*, r1 == r2
end

! CHECK-LABEL: c.func @_QQmain
  use ieee_arithmetic, only: ieee_round_type, ieee_nearest, ieee_to_zero
  interface
    subroutine s(r1,r2)
      import ieee_round_type
      type(ieee_round_type) :: r1, r2
    end
  end interface

  ! CHECK:   %[[V_0:[0-9]+]] = fir.alloca !fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>
  ! CHECK:   %[[V_1:[0-9]+]] = fir.alloca !fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>
  ! CHECK:   %[[V_2:[0-9]+]] = fir.alloca !fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>
  ! CHECK:   %[[V_3:[0-9]+]] = fir.alloca !fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>
  ! CHECK:   %[[V_9:[0-9]+]] = fir.field_index _QMieee_arithmeticTieee_round_type.mode, !fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>
  ! CHECK:   %[[V_10:[0-9]+]] = fir.coordinate_of %[[V_3]], %[[V_9]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:   fir.store %c0{{.*}} to %[[V_10]] : !fir.ref<i8>
  ! CHECK:   %[[V_16:[0-9]+]] = fir.field_index _QMieee_arithmeticTieee_round_type.mode, !fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>
  ! CHECK:   %[[V_17:[0-9]+]] = fir.coordinate_of %[[V_2]], %[[V_16]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:   fir.store %c1{{.*}} to %[[V_17]] : !fir.ref<i8>
  ! CHECK:   fir.call @_QPs(%[[V_3]], %[[V_2]]) {{.*}} : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>, !fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>) -> ()
  call s(ieee_to_zero, ieee_nearest)

  ! CHECK:   %[[V_23:[0-9]+]] = fir.field_index _QMieee_arithmeticTieee_round_type.mode, !fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>
  ! CHECK:   %[[V_24:[0-9]+]] = fir.coordinate_of %[[V_1]], %[[V_23]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:   fir.store %c1{{.*}} to %[[V_24]] : !fir.ref<i8>
  ! CHECK:   %[[V_30:[0-9]+]] = fir.field_index _QMieee_arithmeticTieee_round_type.mode, !fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>
  ! CHECK:   %[[V_31:[0-9]+]] = fir.coordinate_of %[[V_0]], %[[V_30]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:   fir.store %c1{{.*}} to %[[V_31]] : !fir.ref<i8>
  ! CHECK:   fir.call @_QPs(%[[V_1]], %[[V_0]]) {{.*}} : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>, !fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{_QMieee_arithmeticTieee_round_type.mode:i8}>>) -> ()
  call s(ieee_nearest, ieee_nearest)
end

