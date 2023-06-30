! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: c.func @_QQmain
program cs
  use ieee_arithmetic

  ! CHECK-DAG: %[[V_0:[0-9]+]] = fir.alloca f16 {adapt.valuebyref}
  ! CHECK-DAG: %[[V_1:[0-9]+]] = fir.alloca !fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>
  ! CHECK:     %[[V_2:[0-9]+]] = fir.address_of(@_QFEx2) : !fir.ref<f16>
  ! CHECK:     %[[V_3:[0-9]+]] = fir.address_of(@_QFEx4) : !fir.ref<f32>
  ! CHECK:     %[[V_4:[0-9]+]] = fir.address_of(@_QFEy4) : !fir.ref<f32>
  real(2) :: x2 =    2.0
  real(4) :: x4 =    4.0
  real(4) :: y4 = -100.0

  ! CHECK:     %[[V_8:[0-9]+]] = fir.load %[[V_3]] : !fir.ref<f32>
  ! CHECK:     %[[V_9:[0-9]+]] = fir.load %[[V_4]] : !fir.ref<f32>
  ! CHECK:     %[[V_10:[0-9]+]] = llvm.intr.copysign(%[[V_8]], %[[V_9]])  : (f32, f32) -> f32
  ! CHECK:     %[[V_11:[0-9]+]] = fir.call @_FortranAioOutputReal32(%{{.*}}, %[[V_10]]) {{.*}} : (!fir.ref<i8>, f32) -> i1
  print*, ieee_copy_sign(x4, y4)

  ! CHECK:     %[[V_16:[0-9]+]] = fir.load %[[V_2]] : !fir.ref<f16>
  ! CHECK:     %[[V_22:[0-9]+]] = fir.field_index which, !fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>
  ! CHECK:     %[[V_23:[0-9]+]] = fir.coordinate_of %[[V_1]], %[[V_22]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     fir.store %c2{{.*}} to %[[V_23]] : !fir.ref<i8>

  ! CHECK:     %[[V_24:[0-9]+]] = fir.field_index which, !fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>
  ! CHECK:     %[[V_25:[0-9]+]] = fir.coordinate_of %[[V_1]], %[[V_24]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     %[[V_26:[0-9]+]] = fir.load %[[V_25]] : !fir.ref<i8>
  ! CHECK:     %[[V_27:[0-9]+]] = fir.address_of(@_FortranAIeeeValueTable_8) : !fir.ref<!fir.array<12xi64>>
  ! CHECK:     %[[V_28:[0-9]+]] = fir.coordinate_of %[[V_27]], %[[V_26]] : (!fir.ref<!fir.array<12xi64>>, i8) -> !fir.ref<i64>
  ! CHECK:     %[[V_29:[0-9]+]] = fir.load %[[V_28]] : !fir.ref<i64>
  ! CHECK:     %[[V_30:[0-9]+]] = arith.bitcast %[[V_29]] : i64 to f64
  ! CHECK:     %[[V_31:[0-9]+]] = arith.negf %[[V_30]] fastmath<contract> : f64
  ! CHECK:     %[[V_32:[0-9]+]] = arith.bitcast %[[V_16]] : f16 to i16
  ! CHECK:     %[[V_33:[0-9]+]] = arith.bitcast %[[V_31]] : f64 to i64
  ! CHECK:     %[[V_34:[0-9]+]] = arith.shrui %[[V_33]], %c63{{.*}} : i64
  ! CHECK:     %[[V_35:[0-9]+]] = arith.shli %[[V_32]], %c1{{.*}} : i16
  ! CHECK:     %[[V_36:[0-9]+]] = arith.shrui %[[V_35]], %c1{{.*}} : i16
  ! CHECK:     %[[V_37:[0-9]+]] = arith.shli %c1{{.*}}, %c15{{.*}} : i16
  ! CHECK:     %[[V_38:[0-9]+]] = arith.cmpi eq, %[[V_34]], %c0{{.*}} : i64
  ! CHECK:     %[[V_39:[0-9]+]] = arith.select %[[V_38]], %c0{{.*}}, %[[V_37]] : i16
  ! CHECK:     %[[V_40:[0-9]+]] = arith.ori %[[V_36]], %[[V_39]] : i16
  ! CHECK:     %[[V_41:[0-9]+]] = arith.bitcast %[[V_40]] : i16 to f16
  ! CHECK:     fir.store %[[V_41]] to %[[V_0]] : !fir.ref<f16>
  print*, ieee_copy_sign(x2, -ieee_value(0.0_8, ieee_quiet_nan))
end

! CHECK: fir.global linkonce @_FortranAIeeeValueTable_8(dense<[0, 9219994337134247936, 9221120237041090560, -4503599627370496, -4616189618054758400, -9221120237041090560, -9223372036854775808, 0, 2251799813685248, 4607182418800017408, 9218868437227405312, 0]> : tensor<12xi64>) constant : !fir.array<12xi64>

