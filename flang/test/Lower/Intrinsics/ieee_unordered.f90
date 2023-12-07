! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: func @_QQmain
use ieee_arithmetic
! CHECK-DAG: %[[V_0:[0-9]+]] = fir.alloca !fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>
! CHECK-DAG: %[[V_1:[0-9]+]] = fir.alloca !fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>
! CHECK-DAG: %[[V_2:[0-9]+]] = fir.alloca f128 {bindc_name = "x", uniq_name = "_QFEx"}
! CHECK-DAG: %[[V_3:[0-9]+]] = fir.alloca f128 {bindc_name = "y", uniq_name = "_QFEy"}
! CHECK-DAG: %[[V_4:[0-9]+]] = fir.alloca f128 {bindc_name = "z", uniq_name = "_QFEz"}
real(16) :: x, y, z

x = -17.0

! CHECK:     %[[V_10:[0-9]+]] = fir.field_index which, !fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>
! CHECK:     %[[V_11:[0-9]+]] = fir.coordinate_of %[[V_1]], %[[V_10]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>>, !fir.field) -> !fir.ref<i8>
! CHECK:     fir.store %c3{{.*}} to %[[V_11]] : !fir.ref<i8>

! CHECK:     %[[V_12:[0-9]+]] = fir.field_index which, !fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>
! CHECK:     %[[V_13:[0-9]+]] = fir.coordinate_of %[[V_1]], %[[V_12]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>>, !fir.field) -> !fir.ref<i8>
! CHECK:     %[[V_14:[0-9]+]] = fir.load %[[V_13]] : !fir.ref<i8>
! CHECK:     %[[V_15:[0-9]+]] = fir.address_of(@_FortranAIeeeValueTable_16) : !fir.ref<!fir.array<12xi64>>
! CHECK:     %[[V_16:[0-9]+]] = fir.coordinate_of %[[V_15]], %[[V_14]] : (!fir.ref<!fir.array<12xi64>>, i8) -> !fir.ref<i64>
! CHECK:     %[[V_17:[0-9]+]] = fir.load %[[V_16]] : !fir.ref<i64>
! CHECK:     %[[V_18:[0-9]+]] = fir.convert %[[V_17]] : (i64) -> i128
! CHECK:     %[[V_19:[0-9]+]] = arith.shli %[[V_18]], %c64{{.*}} : i128
! CHECK:     %[[V_20:[0-9]+]] = arith.bitcast %[[V_19]] : i128 to f128
! CHECK:     fir.store %[[V_20]] to %[[V_3]] : !fir.ref<f128>
y = ieee_value(y, ieee_negative_inf)

! CHECK:     %[[V_26:[0-9]+]] = fir.field_index which, !fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>
! CHECK:     %[[V_27:[0-9]+]] = fir.coordinate_of %[[V_0]], %[[V_26]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>>, !fir.field) -> !fir.ref<i8>
! CHECK:     fir.store %c2{{.*}} to %[[V_27]] : !fir.ref<i8>
! CHECK:     %[[V_28:[0-9]+]] = fir.field_index which, !fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>
! CHECK:     %[[V_29:[0-9]+]] = fir.coordinate_of %[[V_0]], %[[V_28]] : (!fir.ref<!fir.type<_QMieee_arithmeticTieee_class_type{which:i8}>>, !fir.field) -> !fir.ref<i8>
! CHECK:     %[[V_30:[0-9]+]] = fir.load %[[V_29]] : !fir.ref<i8>
! CHECK:     %[[V_31:[0-9]+]] = fir.address_of(@_FortranAIeeeValueTable_16) : !fir.ref<!fir.array<12xi64>>
! CHECK:     %[[V_32:[0-9]+]] = fir.coordinate_of %[[V_31]], %[[V_30]] : (!fir.ref<!fir.array<12xi64>>, i8) -> !fir.ref<i64>
! CHECK:     %[[V_33:[0-9]+]] = fir.load %[[V_32]] : !fir.ref<i64>
! CHECK:     %[[V_34:[0-9]+]] = fir.convert %[[V_33]] : (i64) -> i128
! CHECK:     %[[V_35:[0-9]+]] = arith.shli %[[V_34]], %c64{{.*}} : i128
! CHECK:     %[[V_36:[0-9]+]] = arith.bitcast %[[V_35]] : i128 to f128
! CHECK:     fir.store %[[V_36]] to %[[V_4]] : !fir.ref<f128>
z = ieee_value(z, ieee_quiet_nan)

! CHECK:     %[[V_40:[0-9]+]] = fir.load %[[V_2]] : !fir.ref<f128>
! CHECK:     %[[V_41:[0-9]+]] = fir.load %[[V_3]] : !fir.ref<f128>
! CHECK:     %[[V_42:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_40]]) <{bit = 3 : i32}> : (f128) -> i1
! CHECK:     %[[V_43:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_41]]) <{bit = 3 : i32}> : (f128) -> i1
! CHECK:     %[[V_44:[0-9]+]] = arith.ori %[[V_42]], %[[V_43]] : i1
! CHECK:     %[[V_45:[0-9]+]] = fir.convert %[[V_44]] : (i1) -> !fir.logical<4>
! CHECK:     %[[V_46:[0-9]+]] = fir.convert %[[V_45]] : (!fir.logical<4>) -> i1
! CHECK:     %[[V_47:[0-9]+]] = fir.call @_FortranAioOutputLogical(%{{.*}}, %[[V_46]]) {{.*}} : (!fir.ref<i8>, i1) -> i1

! CHECK:     %[[V_48:[0-9]+]] = fir.load %[[V_2]] : !fir.ref<f128>
! CHECK:     %[[V_49:[0-9]+]] = fir.load %[[V_4]] : !fir.ref<f128>
! CHECK:     %[[V_50:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_48]]) <{bit = 3 : i32}> : (f128) -> i1
! CHECK:     %[[V_51:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_49]]) <{bit = 3 : i32}> : (f128) -> i1
! CHECK:     %[[V_52:[0-9]+]] = arith.ori %[[V_50]], %[[V_51]] : i1
! CHECK:     %[[V_53:[0-9]+]] = fir.convert %[[V_52]] : (i1) -> !fir.logical<4>
! CHECK:     %[[V_54:[0-9]+]] = fir.convert %[[V_53]] : (!fir.logical<4>) -> i1
! CHECK:     %[[V_55:[0-9]+]] = fir.call @_FortranAioOutputLogical(%{{.*}}, %[[V_54]]) {{.*}} : (!fir.ref<i8>, i1) -> i1

! CHECK:     %[[V_56:[0-9]+]] = fir.load %[[V_3]] : !fir.ref<f128>
! CHECK:     %[[V_57:[0-9]+]] = fir.load %[[V_4]] : !fir.ref<f128>
! CHECK:     %[[V_58:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_56]]) <{bit = 3 : i32}> : (f128) -> i1
! CHECK:     %[[V_59:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_57]]) <{bit = 3 : i32}> : (f128) -> i1
! CHECK:     %[[V_60:[0-9]+]] = arith.ori %[[V_58]], %[[V_59]] : i1
! CHECK:     %[[V_61:[0-9]+]] = fir.convert %[[V_60]] : (i1) -> !fir.logical<4>
! CHECK:     %[[V_62:[0-9]+]] = fir.convert %[[V_61]] : (!fir.logical<4>) -> i1
! CHECK:     %[[V_63:[0-9]+]] = fir.call @_FortranAioOutputLogical(%{{.*}}, %[[V_62]]) {{.*}} : (!fir.ref<i8>, i1) -> i1
print*, ieee_unordered(x,y), ieee_unordered(x,z), ieee_unordered(y,z)
end
