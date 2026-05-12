! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

#ifndef RK
#define RK 8
#endif

module m
  integer, parameter :: k = RK
  character(20) :: tag(11)
contains
  ! CHECK-LABEL: func.func @_QMmPinit()
  subroutine init
    tag( 1) = 'signaling_nan';      tag( 2) = 'quiet_nan'
    tag( 3) = 'negative_inf';       tag( 4) = 'negative_normal'
    tag( 5) = 'negative_denormal';  tag( 6) = 'negative_zero'
    tag( 7) = 'positive_zero';      tag( 8) = 'positive_denormal'
    tag( 9) = 'positive_normal';    tag(10) = 'positive_inf'
    tag(11) = 'other_value'
  end
  ! CHECK-LABEL: func.func @_QMmPout(
  ! CHECK-SAME: %[[X_ARG:.*]]: !fir.ref<f64> {{.*}}, %[[V_ARG:.*]]: !fir.ref<i32> {{.*}})
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

! CHECK-LABEL: func.func @_QPclassify(
! CHECK-SAME: %[[X_ARG:.*]]: !fir.ref<f64> {{.*}})
subroutine classify(x)
  use m; use ieee_arithmetic
  real(k) :: x
  ! CHECK-DAG: %[[R_ALLOC:.*]] = fir.alloca !fir.type<_QMieee_arithmeticTieee_class_type{_QMieee_arithmeticTieee_class_type.which:i8}> {bindc_name = "r", uniq_name = "_QFclassifyEr"}
  ! CHECK-DAG: %[[R_DECL:.*]]:2 = hlfir.declare %[[R_ALLOC]] {uniq_name = "_QFclassifyEr"}
  ! CHECK-DAG: %[[X_DECL:.*]]:2 = hlfir.declare %[[X_ARG]] {{.*}} {uniq_name = "_QFclassifyEx"}
  type(ieee_class_type) :: r

  ! CHECK:     %[[X_VAL:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<f64>
  ! CHECK:     %[[BITCAST:.*]] = arith.bitcast %[[X_VAL]] : f64 to i64
  ! CHECK:     %{{.*}} = arith.shrui %[[BITCAST]], %c59{{.*}} : i64
  ! CHECK:     %[[V_11:.*]] = arith.andi %{{.*}}, %c16{{.*}} : i64
  ! CHECK:     %[[EXP:.*]] = arith.andi %[[BITCAST]], %c{{-?[0-9]+}}{{.*}} : i64
  ! CHECK:     %[[EXP_NZ:.*]] = arith.cmpi ne, %[[EXP]], %c0{{.*}} : i64
  ! CHECK:     %[[V_14:.*]] = arith.select %[[EXP_NZ]], %c8{{.*}}, %c0{{.*}} : i64
  ! CHECK:     %[[V_15:.*]] = arith.ori %[[V_11]], %[[V_14]] : i64
  ! CHECK:     %[[EXP_INF:.*]] = arith.cmpi eq, %[[EXP]], %c{{-?[0-9]+}}{{.*}} : i64
  ! CHECK:     %[[V_17:.*]] = arith.select %[[EXP_INF]], %c4{{.*}}, %c0{{.*}} : i64
  ! CHECK:     %[[V_18:.*]] = arith.ori %[[V_15]], %[[V_17]] : i64
  ! CHECK:     %[[FRAC:.*]] = arith.andi %[[BITCAST]], %c{{[0-9]+}}{{.*}} : i64
  ! CHECK:     %[[FRAC_NZ:.*]] = arith.cmpi ne, %[[FRAC]], %c0{{.*}} : i64
  ! CHECK:     %[[V_21:.*]] = arith.select %[[FRAC_NZ]], %c2{{.*}}, %c0{{.*}} : i64
  ! CHECK:     %[[V_22:.*]] = arith.ori %[[V_18]], %[[V_21]] : i64
  ! CHECK:     %[[V_23:.*]] = arith.shrui %[[BITCAST]], %c51{{.*}} : i64
  ! CHECK:     %[[V_24:.*]] = arith.andi %[[V_23]], %c1{{.*}} : i64
  ! CHECK:     %[[V_25:.*]] = arith.ori %[[V_22]], %[[V_24]] : i64
  ! CHECK:     %[[TABLE:.*]] = fir.address_of(@_FortranAIeeeClassTable) : !fir.ref<!fir.array<32xi8>>
  ! CHECK:     %[[COORD:.*]] = fir.coordinate_of %[[TABLE]], %[[V_25]] : (!fir.ref<!fir.array<32xi8>>, i64) -> !fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{_QMieee_arithmeticTieee_class_type.which:i8}>>
  ! CHECK:     %[[TMP:.*]]:2 = hlfir.declare %[[COORD]] {uniq_name = ".tmp.intrinsic_result"}
  ! CHECK:     %[[EXPR:.*]] = hlfir.as_expr %[[TMP]]#0 move {{.*}} : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{_QMieee_arithmeticTieee_class_type.which:i8}>>, i1) -> !hlfir.expr<!fir.type<_QMieee_arithmeticTieee_class_type{_QMieee_arithmeticTieee_class_type.which:i8}>>
  ! CHECK:     hlfir.assign %[[EXPR]] to %[[R_DECL]]#0 : !hlfir.expr<!fir.type<_QMieee_arithmeticTieee_class_type{_QMieee_arithmeticTieee_class_type.which:i8}>>, !fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{_QMieee_arithmeticTieee_class_type.which:i8}>>
  ! CHECK:     hlfir.destroy %[[EXPR]] : !hlfir.expr<!fir.type<_QMieee_arithmeticTieee_class_type{_QMieee_arithmeticTieee_class_type.which:i8}>>
  r = ieee_class(x)

! if (r==ieee_signaling_nan)      call out(x, 1)
! if (r==ieee_quiet_nan)          call out(x, 2)
  ! CHECK:     %[[R_WHICH:.*]] = fir.coordinate_of %[[R_DECL]]#0, {{.*}}which : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{_QMieee_arithmeticTieee_class_type.which:i8}>>) -> !fir.ref<i8>
  ! CHECK:     %{{.*}} = fir.coordinate_of %{{.*}}, {{.*}}which : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{_QMieee_arithmeticTieee_class_type.which:i8}>>) -> !fir.ref<i8>
  ! CHECK:     %[[R_VAL:.*]] = fir.load %[[R_WHICH]] : !fir.ref<i8>
  ! CHECK:     %[[CLASS_VAL:.*]] = fir.load %{{.*}} : !fir.ref<i8>
  ! CHECK:     %[[EQ:.*]] = arith.cmpi eq, %[[R_VAL]], %[[CLASS_VAL]] : i8
  ! CHECK:     fir.if %[[EQ]] {
  ! CHECK:       %[[V_ASSOC:.*]]:3 = hlfir.associate %c3{{.*}} {adapt.valuebyref} : (i32) -> (!fir.ref<i32>, !fir.ref<i32>, i1)
  ! CHECK:       fir.call @_QMmPout(%[[X_DECL]]#0, %[[V_ASSOC]]#0) {{.*}} : (!fir.ref<f64>, !fir.ref<i32>) -> ()
  ! CHECK:       hlfir.end_associate %[[V_ASSOC]]#1, %[[V_ASSOC]]#2 : !fir.ref<i32>, i1
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

! CHECK-LABEL: func.func @_QQmain()
program p
  use m; use ieee_arithmetic
  real(k) :: x(10)

  call init

! x(1)  = ieee_value(x(1), ieee_signaling_nan)
! x(2)  = ieee_value(x(1), ieee_quiet_nan)
  ! CHECK:     %[[X_G_ALLOC:.*]] = fir.address_of(@_QFEx) : !fir.ref<!fir.array<10xf64>>
  ! CHECK:     %[[X_G_DECL:.*]]:2 = hlfir.declare %[[X_G_ALLOC]]({{.*}}) {uniq_name = "_QFEx"}
  ! CHECK:     %[[VAL_TABLE:.*]] = fir.address_of(@_FortranAIeeeValueTable_8) : !fir.ref<!fir.array<12xi64>>
  ! CHECK:     %[[VAL_COORD:.*]] = fir.coordinate_of %[[VAL_TABLE]], %{{.*}} : (!fir.ref<!fir.array<12xi64>>, i8) -> !fir.ref<i64>
  ! CHECK:     %[[VAL_I64:.*]] = fir.load %[[VAL_COORD]] : !fir.ref<i64>
  ! CHECK:     %[[VAL_F64:.*]] = arith.bitcast %[[VAL_I64]] : i64 to f64
  ! CHECK:     %[[X3_ADDR:.*]] = hlfir.designate %[[X_G_DECL]]#0 (%c3{{.*}})  : (!fir.ref<!fir.array<10xf64>>, index) -> !fir.ref<f64>
  ! CHECK:     hlfir.assign %[[VAL_F64]] to %[[X3_ADDR]] : f64, !fir.ref<f64>
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
