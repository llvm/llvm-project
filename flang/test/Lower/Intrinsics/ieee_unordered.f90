! REQUIRES: flang-supports-f128-math
! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! CHECK-LABEL: func @_QQmain
use ieee_arithmetic
! CHECK-DAG: %[[V_X:[0-9]+]] = fir.alloca f128 {bindc_name = "x", uniq_name = "_QFEx"}
! CHECK-DAG: %[[X_DECL:[0-9]+]]:2 = hlfir.declare %[[V_X]] {uniq_name = "_QFEx"}
! CHECK-DAG: %[[V_Y:[0-9]+]] = fir.alloca f128 {bindc_name = "y", uniq_name = "_QFEy"}
! CHECK-DAG: %[[Y_DECL:[0-9]+]]:2 = hlfir.declare %[[V_Y]] {uniq_name = "_QFEy"}
! CHECK-DAG: %[[V_Z:[0-9]+]] = fir.alloca f128 {bindc_name = "z", uniq_name = "_QFEz"}
! CHECK-DAG: %[[Z_DECL:[0-9]+]]:2 = hlfir.declare %[[V_Z]] {uniq_name = "_QFEz"}
real(16) :: x, y, z

x = -17.0

! CHECK:     %[[V_NEG_INF:[0-9]+]]:2 = hlfir.declare %{{.*}} {{{.*}}uniq_name = "_QQro._QMieee_arithmeticTieee_class_type.{{[0-9]+}}"}
! CHECK:     %[[V_11:[0-9]+]] = fir.coordinate_of %[[V_NEG_INF]]#0, _QMieee_arithmeticTieee_class_type.which : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{_QMieee_arithmeticTieee_class_type.which:i8}>>) -> !fir.ref<i8>
! CHECK:     %[[V_12:[0-9]+]] = fir.load %[[V_11]] : !fir.ref<i8>
! CHECK:     %[[V_13:[0-9]+]] = fir.address_of(@_FortranAIeeeValueTable_16) : !fir.ref<!fir.array<12xi64>>
! CHECK:     %[[V_14:[0-9]+]] = fir.coordinate_of %[[V_13]], %[[V_12]] : (!fir.ref<!fir.array<12xi64>>, i8) -> !fir.ref<i64>
! CHECK:     %[[V_15:[0-9]+]] = fir.load %[[V_14]] : !fir.ref<i64>
! CHECK:     %[[V_16:[0-9]+]] = fir.convert %[[V_15]] : (i64) -> i128
! CHECK:     %[[V_17:[0-9]+]] = arith.shli %[[V_16]], %c64{{.*}} : i128
! CHECK:     %[[V_18:[0-9]+]] = arith.bitcast %[[V_17]] : i128 to f128
! CHECK:     hlfir.assign %[[V_18]] to %[[Y_DECL]]#0 : f128, !fir.ref<f128>
y = ieee_value(y, ieee_negative_inf)

! CHECK:     %[[V_QNAN:[0-9]+]]:2 = hlfir.declare %{{.*}} {{{.*}}uniq_name = "_QQro._QMieee_arithmeticTieee_class_type.{{[0-9]+}}"}
! CHECK:     %[[V_27:[0-9]+]] = fir.coordinate_of %[[V_QNAN]]#0, _QMieee_arithmeticTieee_class_type.which : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{_QMieee_arithmeticTieee_class_type.which:i8}>>) -> !fir.ref<i8>
! CHECK:     %[[V_28:[0-9]+]] = fir.load %[[V_27]] : !fir.ref<i8>
! CHECK:     %[[V_29:[0-9]+]] = fir.address_of(@_FortranAIeeeValueTable_16) : !fir.ref<!fir.array<12xi64>>
! CHECK:     %[[V_30:[0-9]+]] = fir.coordinate_of %[[V_29]], %[[V_28]] : (!fir.ref<!fir.array<12xi64>>, i8) -> !fir.ref<i64>
! CHECK:     %[[V_31:[0-9]+]] = fir.load %[[V_30]] : !fir.ref<i64>
! CHECK:     %[[V_32:[0-9]+]] = fir.convert %[[V_31]] : (i64) -> i128
! CHECK:     %[[V_33:[0-9]+]] = arith.shli %[[V_32]], %c64{{.*}} : i128
! CHECK:     %[[V_34:[0-9]+]] = arith.bitcast %[[V_33]] : i128 to f128
! CHECK:     hlfir.assign %[[V_34]] to %[[Z_DECL]]#0 : f128, !fir.ref<f128>
z = ieee_value(z, ieee_quiet_nan)

! CHECK:     %[[V_40:[0-9]+]] = fir.load %[[X_DECL]]#0 : !fir.ref<f128>
! CHECK:     %[[V_41:[0-9]+]] = fir.load %[[Y_DECL]]#0 : !fir.ref<f128>
! CHECK:     %[[V_44:[0-9]+]] = arith.cmpf uno, %[[V_40]], %[[V_41]] {{.*}} : f128
! CHECK:     %[[V_45:[0-9]+]] = fir.convert %[[V_44]] : (i1) -> !fir.logical<4>
! CHECK:     %[[V_46:[0-9]+]] = fir.convert %[[V_45]] : (!fir.logical<4>) -> i1
! CHECK:     %[[V_47:[0-9]+]] = fir.call @_FortranAioOutputLogical(%{{.*}}, %[[V_46]]) {{.*}} : (!fir.ref<i8>, i1) -> i1

! CHECK:     %[[V_48:[0-9]+]] = fir.load %[[X_DECL]]#0 : !fir.ref<f128>
! CHECK:     %[[V_49:[0-9]+]] = fir.load %[[Z_DECL]]#0 : !fir.ref<f128>
! CHECK:     %[[V_52:[0-9]+]] = arith.cmpf uno, %[[V_48]], %[[V_49]] {{.*}} : f128
! CHECK:     %[[V_53:[0-9]+]] = fir.convert %[[V_52]] : (i1) -> !fir.logical<4>
! CHECK:     %[[V_54:[0-9]+]] = fir.convert %[[V_53]] : (!fir.logical<4>) -> i1
! CHECK:     %[[V_55:[0-9]+]] = fir.call @_FortranAioOutputLogical(%{{.*}}, %[[V_54]]) {{.*}} : (!fir.ref<i8>, i1) -> i1

! CHECK:     %[[V_56:[0-9]+]] = fir.load %[[Y_DECL]]#0 : !fir.ref<f128>
! CHECK:     %[[V_57:[0-9]+]] = fir.load %[[Z_DECL]]#0 : !fir.ref<f128>
! CHECK:     %[[V_60:[0-9]+]] = arith.cmpf uno, %[[V_56]], %[[V_57]] {{.*}} : f128
! CHECK:     %[[V_61:[0-9]+]] = fir.convert %[[V_60]] : (i1) -> !fir.logical<4>
! CHECK:     %[[V_62:[0-9]+]] = fir.convert %[[V_61]] : (!fir.logical<4>) -> i1
! CHECK:     %[[V_63:[0-9]+]] = fir.call @_FortranAioOutputLogical(%{{.*}}, %[[V_62]]) {{.*}} : (!fir.ref<i8>, i1) -> i1
print*, ieee_unordered(x,y), ieee_unordered(x,z), ieee_unordered(y,z)
end
