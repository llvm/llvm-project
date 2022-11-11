! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: @_QPs
subroutine s(r1,r2)
  use ieee_arithmetic, only: ieee_round_type, operator(==)
  type(ieee_round_type) :: r1, r2
  ! CHECK:   %[[V_3:[0-9]+]] = fir.field_index mode, !fir.type<_QMieee_arithmeticTieee_round_type{mode:i8}>
  ! CHECK:   %[[V_4:[0-9]+]] = fir.coordinate_of %arg0, %[[V_3]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:   %[[V_5:[0-9]+]] = fir.load %[[V_4]] : !fir.ref<i8>
  ! CHECK:   %[[V_6:[0-9]+]] = fir.coordinate_of %arg1, %[[V_3]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:   %[[V_7:[0-9]+]] = fir.load %[[V_6]] : !fir.ref<i8>
  ! CHECK:   %[[V_8:[0-9]+]] = arith.cmpi eq, %[[V_5]], %[[V_7]] : i8
  ! CHECK:   %[[V_9:[0-9]+]] = fir.call @_FortranAioOutputLogical(%{{.*}}, %[[V_8]]) : (!fir.ref<i8>, i1) -> i1
  print*, r1 == r2
end

! CHECK-LABEL: @_QQmain
  use ieee_arithmetic, only: ieee_round_type, ieee_nearest, ieee_to_zero
  interface
    subroutine s(r1,r2)
      import ieee_round_type
      type(ieee_round_type) :: r1, r2
    end
  end interface
  ! CHECK:   %[[V_0:[0-9]+]] = fir.alloca !fir.type<_QMieee_arithmeticTieee_round_type{mode:i8}>
  ! CHECK:   %[[V_1:[0-9]+]] = fir.alloca !fir.type<_QMieee_arithmeticTieee_round_type{mode:i8}>
  ! CHECK:   %[[V_2:[0-9]+]] = fir.alloca !fir.type<_QMieee_arithmeticTieee_round_type{mode:i8}>
  ! CHECK:   %[[V_3:[0-9]+]] = fir.alloca !fir.type<_QMieee_arithmeticTieee_round_type{mode:i8}>
  ! CHECK:   %[[V_4:[0-9]+]] = fir.field_index mode, !fir.type<_QMieee_arithmeticTieee_round_type{mode:i8}>
  ! CHECK:   %[[V_5:[0-9]+]] = fir.coordinate_of %[[V_3]], %[[V_4]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:   fir.store %c2{{.*}} to %[[V_5]] : !fir.ref<i8>
  ! CHECK:   %[[V_6:[0-9]+]] = fir.field_index mode, !fir.type<_QMieee_arithmeticTieee_round_type{mode:i8}>
  ! CHECK:   %[[V_7:[0-9]+]] = fir.coordinate_of %[[V_2]], %[[V_6]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:   fir.store %c1{{.*}} to %[[V_7]] : !fir.ref<i8>
  call s(ieee_to_zero, ieee_nearest)

  ! CHECK:   fir.call @_QPs(%[[V_3]], %[[V_2]]) : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{mode:i8}>>, !fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{mode:i8}>>) -> ()
  ! CHECK:   %[[V_8:[0-9]+]] = fir.field_index mode, !fir.type<_QMieee_arithmeticTieee_round_type{mode:i8}>
  ! CHECK:   %[[V_9:[0-9]+]] = fir.coordinate_of %[[V_1]], %[[V_8]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:   fir.store %c1{{.*}} to %[[V_9]] : !fir.ref<i8>
  ! CHECK:   %[[V_10:[0-9]+]] = fir.field_index mode, !fir.type<_QMieee_arithmeticTieee_round_type{mode:i8}>
  ! CHECK:   %[[V_11:[0-9]+]] = fir.coordinate_of %[[V_0]], %[[V_10]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{mode:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:   fir.store %c1{{.*}} to %[[V_11]] : !fir.ref<i8>
  ! CHECK:   fir.call @_QPs(%[[V_1]], %[[V_0]]) : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{mode:i8}>>, !fir.ref<!fir.type<_QMieee_arithmeticTieee_round_type{mode:i8}>>) -> ()
  call s(ieee_nearest, ieee_nearest)
end
