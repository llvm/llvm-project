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
  ! CHECK:     %[[V_27:[0-9]+]] = fir.address_of(@_FortranAIeeeValueTable_16) : !fir.ref<!fir.array<12xi64>>
  ! CHECK:     %[[V_28:[0-9]+]] = fir.coordinate_of %[[V_27]], %[[V_26]] : (!fir.ref<!fir.array<12xi64>>, i8) -> !fir.ref<i64>
  ! CHECK:     %[[V_29:[0-9]+]] = fir.load %[[V_28]] : !fir.ref<i64>
  ! CHECK:     %[[V_30:[0-9]+]] = fir.convert %[[V_29]] : (i64) -> i128
  ! CHECK:     %[[V_31:[0-9]+]] = arith.shli %[[V_30]], %c64{{.*}} : i128
  ! CHECK:     %[[V_32:[0-9]+]] = arith.bitcast %[[V_31]] : i128 to f128
  ! CHECK:     %[[V_33:[0-9]+]] = arith.negf %[[V_32]] {{.*}} : f128
  ! CHECK:     %[[V_34:[0-9]+]] = arith.bitcast %[[V_16]] : f16 to i16
  ! CHECK:     %[[V_35:[0-9]+]] = arith.bitcast %[[V_33]] : f128 to i128
  ! CHECK:     %[[V_36:[0-9]+]] = arith.shrui %[[V_35]], %c127{{.*}} : i128
  ! CHECK:     %[[V_37:[0-9]+]] = arith.shli %[[V_34]], %c1{{.*}} : i16
  ! CHECK:     %[[V_38:[0-9]+]] = arith.shrui %[[V_37]], %c1{{.*}} : i16
  ! CHECK:     %[[V_39:[0-9]+]] = arith.shli %c1{{.*}}, %c15{{.*}} : i16
  ! CHECK:     %[[V_40:[0-9]+]] = arith.cmpi eq, %[[V_36]], %c0{{.*}} : i128
  ! CHECK:     %[[V_41:[0-9]+]] = arith.select %[[V_40]], %c0{{.*}}, %[[V_39]] : i16
  ! CHECK:     %[[V_42:[0-9]+]] = arith.ori %[[V_38]], %[[V_41]] : i16
  ! CHECK:     %[[V_43:[0-9]+]] = arith.bitcast %[[V_42]] : i16 to f16
  ! CHECK:     fir.store %[[V_43]] to %[[V_0]] : !fir.ref<f16>
  print*, ieee_copy_sign(x2, -ieee_value(0.0_16, ieee_quiet_nan))
end

! CHECK: fir.global linkonce @_FortranAIeeeValueTable_16(dense<[0, 9223160930622242816, 9223231299366420480, -281474976710656, -4611967493404098560, -9223336852482686976, -9223372036854775808, 0, 35184372088832, 4611404543450677248, 9223090561878065152, 0]> : tensor<12xi64>) constant : !fir.array<12xi64>
