! RUN: bbc -emit-fir %s -o - | FileCheck %s

#ifndef RK
#define RK 8
#endif

module m
  integer, parameter :: k = RK
  character(20) :: tag(11)
contains
  ! CHECK-LABEL: func @_QMmPinit
  subroutine init
    tag( 1) = 'signaling_nan';      tag( 2) = 'quiet_nan'
    tag( 3) = 'negative_inf';       tag( 4) = 'negative_normal'
    tag( 5) = 'negative_denormal';  tag( 6) = 'negative_zero'
    tag( 7) = 'positive_zero';      tag( 8) = 'positive_denormal'
    tag( 9) = 'positive_normal';    tag(10) = 'positive_inf'
    tag(11) = 'other_value'
  end
  ! CHECK-LABEL: func @_QMmPout
  subroutine out(x,v)
    use ieee_arithmetic
    real(k) :: x
    integer :: v
    logical :: L(4)
    L(1) = ieee_is_finite(x)
    L(2) = ieee_is_nan(x)
    L(3) = ieee_is_negative(x)
    L(4) = ieee_is_normal(x)
!   if (k== 2) print "('  k=2 ',f7.2,z6.4,  i4,': ',a18,4L2)", x,x, v, tag(v), L
!   if (k== 3) print "('  k=3 ',f7.2,z6.4,  i4,': ',a18,4L2)", x,x, v, tag(v), L
!   if (k== 4) print "('  k=4 ',f7.2,z10.8, i4,': ',a18,4L2)", x,x, v, tag(v), L
    if (k== 8) print "('  k=8 ',f7.2,z18.16,i4,': ',a18,4L2)", x,x, v, tag(v), L
!   if (k==10) print "('  k=10',f7.2,z22.20,i4,': ',a18,4L2)", x,x, v, tag(v), L
!   if (k==16) print "('  k=16',f7.2,z34.32,i4,': ',a18,4L2)", x,x, v, tag(v), L
  end
end module m

! CHECK-LABEL: func @_QPclassify
subroutine classify(x)
  use m; use ieee_arithmetic
  real(k) :: x
  ! CHECK-DAG: %[[V_0:[0-9]+]] = fir.alloca i32 {adapt.valuebyref}
  ! CHECK-DAG: %[[V_1:[0-9]+]] = fir.alloca !fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>
  ! CHECK-DAG: %[[V_2:[0-9]+]] = fir.alloca !fir.type<_QMieee_arithmeticTieee_class_type{which:i8}> {bindc_name = "r", uniq_name = "_QFclassifyEr"}
  type(ieee_class_type) :: r

  ! CHECK:     %[[V_8:[0-9]+]] = fir.load %arg0 : !fir.ref<f64>
  ! CHECK:     %[[V_9:[0-9]+]] = arith.bitcast %[[V_8]] : f64 to i64
  ! CHECK:     %[[V_10:[0-9]+]] = arith.shrui %[[V_9]], %c59{{.*}} : i64
  ! CHECK:     %[[V_11:[0-9]+]] = arith.andi %[[V_10]], %c16{{.*}} : i64
  ! CHECK:     %[[V_12:[0-9]+]] = arith.andi %[[V_9]], %c9218868437227405312{{.*}} : i64
  ! CHECK:     %[[V_13:[0-9]+]] = arith.cmpi ne, %[[V_12]], %c0{{.*}} : i64
  ! CHECK:     %[[V_14:[0-9]+]] = arith.select %[[V_13]], %c8{{.*}}, %c0{{.*}} : i64
  ! CHECK:     %[[V_15:[0-9]+]] = arith.ori %[[V_11]], %[[V_14]] : i64
  ! CHECK:     %[[V_16:[0-9]+]] = arith.cmpi eq, %[[V_12]], %c9218868437227405312{{.*}} : i64
  ! CHECK:     %[[V_17:[0-9]+]] = arith.select %[[V_16]], %c4{{.*}}, %c0{{.*}} : i64
  ! CHECK:     %[[V_18:[0-9]+]] = arith.ori %[[V_15]], %[[V_17]] : i64
  ! CHECK:     %[[V_19:[0-9]+]] = arith.andi %[[V_9]], %c2251799813685247{{.*}} : i64
  ! CHECK:     %[[V_20:[0-9]+]] = arith.cmpi ne, %[[V_19]], %c0{{.*}} : i64
  ! CHECK:     %[[V_21:[0-9]+]] = arith.select %[[V_20]], %c2{{.*}}, %c0{{.*}} : i64
  ! CHECK:     %[[V_22:[0-9]+]] = arith.ori %[[V_18]], %[[V_21]] : i64
  ! CHECK:     %[[V_23:[0-9]+]] = arith.shrui %[[V_9]], %c51{{.*}} : i64
  ! CHECK:     %[[V_24:[0-9]+]] = arith.andi %[[V_23]], %c1{{.*}} : i64
  ! CHECK:     %[[V_25:[0-9]+]] = arith.ori %[[V_22]], %[[V_24]] : i64
  ! CHECK:     %[[V_26:[0-9]+]] = fir.address_of(@_FortranAIeeeClassTable) : !fir.ref<!fir.array<32xi8>>
  ! CHECK:     %[[V_27:[0-9]+]] = fir.coordinate_of %[[V_26]], %[[V_25]] : (!fir.ref<!fir.array<32xi8>>, i64) -> !fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>>
  ! CHECK:     %[[V_28:[0-9]+]] = fir.field_index which, !fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>
  ! CHECK:     %[[V_29:[0-9]+]] = fir.coordinate_of %[[V_27]], %[[V_28]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     %[[V_30:[0-9]+]] = fir.field_index which, !fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>
  ! CHECK:     %[[V_31:[0-9]+]] = fir.coordinate_of %[[V_2]], %[[V_30]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     %[[V_32:[0-9]+]] = fir.load %[[V_29]] : !fir.ref<i8>
  ! CHECK:     fir.store %[[V_32]] to %[[V_31]] : !fir.ref<i8>
  r = ieee_class(x)

! if (r==ieee_signaling_nan)      call out(x, 1)
! if (r==ieee_quiet_nan)          call out(x, 2)
  ! CHECK:     %[[V_38:[0-9]+]] = fir.field_index which, !fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>
  ! CHECK:     %[[V_39:[0-9]+]] = fir.coordinate_of %[[V_1]], %[[V_38]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     fir.store %c3{{.*}} to %[[V_39]] : !fir.ref<i8>
  ! CHECK:     %[[V_40:[0-9]+]] = fir.field_index which, !fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>
  ! CHECK:     %[[V_41:[0-9]+]] = fir.coordinate_of %[[V_2]], %[[V_40]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     %[[V_42:[0-9]+]] = fir.field_index which, !fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>
  ! CHECK:     %[[V_43:[0-9]+]] = fir.coordinate_of %[[V_1]], %[[V_42]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     %[[V_44:[0-9]+]] = fir.load %[[V_41]] : !fir.ref<i8>
  ! CHECK:     %[[V_45:[0-9]+]] = fir.load %[[V_43]] : !fir.ref<i8>
  ! CHECK:     %[[V_46:[0-9]+]] = arith.cmpi eq, %[[V_44]], %[[V_45]] : i8
  ! CHECK:     fir.if %[[V_46]] {
  ! CHECK:       fir.store %c3{{.*}} to %[[V_0]] : !fir.ref<i32>
  ! CHECK:       fir.call @_QMmPout(%arg0, %[[V_0]]) {{.*}} : (!fir.ref<f64>, !fir.ref<i32>) -> ()
  ! CHECK:     }
  if (r==ieee_negative_inf)       call out(x, 3)
! if (r==ieee_negative_normal)    call out(x, 4)
! if (r==ieee_negative_denormal)  call out(x, 5)
! if (r==ieee_negative_zero)      call out(x, 6)
! if (r==ieee_positive_zero)      call out(x, 7)
! if (r==ieee_positive_denormal)  call out(x, 8)
! if (r==ieee_positive_normal)    call out(x, 9)
! if (r==ieee_positive_inf)       call out(x,10)
! if (r==ieee_other_value)        call out(x,11)
end

! CHECK-LABEL: func @_QQmain
program p
  use m; use ieee_arithmetic
  real(k) :: x(10)

  call init

! x(1)  = ieee_value(x(1), ieee_signaling_nan)
! x(2)  = ieee_value(x(1), ieee_quiet_nan)
  ! CHECK:     %[[V_0:[0-9]+]] = fir.alloca !fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>
  ! CHECK:     %[[V_2:[0-9]+]] = fir.address_of(@_QFEx) : !fir.ref<!fir.array<10xf64>>
  ! CHECK:     %[[V_8:[0-9]+]] = fir.field_index which, !fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>
  ! CHECK:     %[[V_9:[0-9]+]] = fir.coordinate_of %[[V_0]], %[[V_8]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     fir.store %c3{{.*}} to %[[V_9]] : !fir.ref<i8>
  ! CHECK:     %[[V_10:[0-9]+]] = fir.field_index which, !fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>
  ! CHECK:     %[[V_11:[0-9]+]] = fir.coordinate_of %[[V_0]], %[[V_10]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>>, !fir.field) -> !fir.ref<i8>
  ! CHECK:     %[[V_12:[0-9]+]] = fir.load %[[V_11]] : !fir.ref<i8>
  ! CHECK:     %[[V_13:[0-9]+]] = fir.address_of(@_FortranAIeeeValueTable_8) : !fir.ref<!fir.array<12xi64>>
  ! CHECK:     %[[V_14:[0-9]+]] = fir.coordinate_of %[[V_13]], %[[V_12]] : (!fir.ref<!fir.array<12xi64>>, i8) -> !fir.ref<i64>
  ! CHECK:     %[[V_15:[0-9]+]] = fir.load %[[V_14]] : !fir.ref<i64>
  ! CHECK:     %[[V_16:[0-9]+]] = arith.bitcast %[[V_15]] : i64 to f64
  ! CHECK:     %[[V_17:[0-9]+]] = arith.subi %c3{{.*}}, %c1{{.*}} : i64
  ! CHECK:     %[[V_18:[0-9]+]] = fir.coordinate_of %[[V_2]], %[[V_17]] : (!fir.ref<!fir.array<10xf64>>, i64) -> !fir.ref<f64>
  ! CHECK:     fir.store %[[V_16]] to %[[V_18]] : !fir.ref<f64>
  x(3)  = ieee_value(x(1), ieee_negative_inf)
! x(4)  = ieee_value(x(1), ieee_negative_normal)
! x(5)  = ieee_value(x(1), ieee_negative_subnormal)
! x(6)  = ieee_value(x(1), ieee_negative_zero)
! x(7)  = ieee_value(x(1), ieee_positive_zero)
! x(8)  = ieee_value(x(1), ieee_positive_subnormal)
! x(9)  = ieee_value(x(1), ieee_positive_normal)
! x(10) = ieee_value(x(1), ieee_positive_inf)

  do i = 1,10
    call classify(x(i))
  enddo
end

! CHECK: fir.global linkonce @_FortranAIeeeClassTable(dense<[7, 8, 8, 8, 11, 11, 11, 11, 9, 9, 9, 9, 10, 2, 1, 2, 6, 5, 5, 5, 11, 11, 11, 11, 4, 4, 4, 4, 3, 2, 1, 2]> : tensor<32xi8>) constant : !fir.array<32xi8>
! CHECK: fir.global linkonce @_FortranAIeeeValueTable_8(dense<[0, 9219994337134247936, 9221120237041090560, -4503599627370496, -4616189618054758400, -9221120237041090560, -9223372036854775808, 0, 2251799813685248, 4607182418800017408, 9218868437227405312, 0]> : tensor<12xi64>) constant : !fir.array<12xi64>
