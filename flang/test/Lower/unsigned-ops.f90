! RUN: %flang_fc1 -funsigned -emit-mlir %s -o - | FileCheck %s

unsigned function f01(u, v)
  unsigned, intent(in) :: u, v
  f01 = u + v - 1u
end

!CHECK: func.func @_QPf01(%[[ARG0:.*]]: !fir.ref<ui32> {fir.bindc_name = "u", fir.fortran_attrs = #fir.var_attrs<intent_in>, fir.read_only}, %[[ARG1:.*]]: !fir.ref<ui32> {fir.bindc_name = "v", fir.fortran_attrs = #fir.var_attrs<intent_in>, fir.read_only}) -> ui32 {
!CHECK: %[[C1_I32:.*]] = arith.constant 1 : i32
!CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
!CHECK: %[[VAL_1:.*]] = fir.alloca ui32 <{bindc_name = "f01", uniq_name = "_QFf01Ef01"}>
!CHECK: %[[VAL_2:.*]] = fir.declare %[[VAL_1]] uniq_name("_QFf01Ef01") : (!fir.ref<ui32>) -> !fir.ref<ui32>
!CHECK: %[[VAL_3:.*]] = fir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg {{[0-9]+}} uniq_name("_QFf01Eu") fortran_attrs<intent_in> : (!fir.ref<ui32>, !fir.dscope) -> !fir.ref<ui32>
!CHECK: %[[VAL_4:.*]] = fir.declare %[[ARG1]] dummy_scope %[[VAL_0]] arg {{[0-9]+}} uniq_name("_QFf01Ev") fortran_attrs<intent_in> : (!fir.ref<ui32>, !fir.dscope) -> !fir.ref<ui32>
!CHECK: %[[VAL_5:.*]] = fir.load %[[VAL_3]] : !fir.ref<ui32>
!CHECK: %[[VAL_6:.*]] = fir.load %[[VAL_4]] : !fir.ref<ui32>
!CHECK: %[[VAL_7:.*]] = fir.convert %[[VAL_5]] : (ui32) -> i32
!CHECK: %[[VAL_8:.*]] = fir.convert %[[VAL_6]] : (ui32) -> i32
!CHECK: %[[VAL_9:.*]] = arith.addi %[[VAL_7]], %[[VAL_8]] : i32
!CHECK: %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i32) -> ui32
!CHECK: %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (ui32) -> i32
!CHECK: %[[VAL_12:.*]] = arith.subi %[[VAL_11]], %[[C1_I32]] : i32
!CHECK: %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> ui32
!CHECK: fir.store %[[VAL_13]] to %[[VAL_2]] : !fir.ref<ui32>
!CHECK: %[[VAL_14:.*]] = fir.load %[[VAL_2]] : !fir.ref<ui32>
!CHECK: return %[[VAL_14]] : ui32

unsigned function f02(u, v)
  unsigned, intent(in) :: u, v
  f02 = u ** v - 1u
end

!CHECK: func.func @_QPf02(%[[ARG0:.*]]: !fir.ref<ui32> {fir.bindc_name = "u", fir.fortran_attrs = #fir.var_attrs<intent_in>, fir.read_only}, %[[ARG1:.*]]: !fir.ref<ui32> {fir.bindc_name = "v", fir.fortran_attrs = #fir.var_attrs<intent_in>, fir.read_only}) -> ui32 {
!CHECK: %[[C1_i32:.*]] = arith.constant 1 : i32
!CHECK: %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
!CHECK: %[[VAL_1:.*]] = fir.alloca ui32 <{bindc_name = "f02", uniq_name = "_QFf02Ef02"}>
!CHECK: %[[VAL_2:.*]] = fir.declare %[[VAL_1]] uniq_name("_QFf02Ef02") : (!fir.ref<ui32>) -> !fir.ref<ui32>
!CHECK: %[[VAL_3:.*]] = fir.declare %[[ARG0]] dummy_scope %[[VAL_0]] arg {{[0-9]+}} uniq_name("_QFf02Eu") fortran_attrs<intent_in> : (!fir.ref<ui32>, !fir.dscope) -> !fir.ref<ui32>
!CHECK: %[[VAL_4:.*]] = fir.declare %[[ARG1]] dummy_scope %[[VAL_0]] arg {{[0-9]+}} uniq_name("_QFf02Ev") fortran_attrs<intent_in> : (!fir.ref<ui32>, !fir.dscope) -> !fir.ref<ui32>
!CHECK: %[[VAL_5:.*]] = fir.load %[[VAL_3]] : !fir.ref<ui32>
!CHECK: %[[VAL_6:.*]] = fir.load %[[VAL_4]] : !fir.ref<ui32>
!CHECK: %[[VAL_7:.*]] = fir.convert %[[VAL_5]] : (ui32) -> i64
!CHECK: %[[VAL_8:.*]] = fir.convert %[[VAL_6]] : (ui32) -> i64
!CHECK: %[[VAL_9:.*]] = fir.call @_FortranAUPow8(%[[VAL_7]], %[[VAL_8]]) fastmath<contract> : (i64, i64) -> i64
!CHECK: %[[VAL_10:.*]] = fir.convert %[[VAL_9]] : (i64) -> ui32
!CHECK: %[[VAL_11:.*]] = fir.convert %[[VAL_10]] : (ui32) -> i32
!CHECK: %[[VAL_12:.*]] = arith.subi %[[VAL_11]], %[[C1_i32]] : i32
!CHECK: %[[VAL_13:.*]] = fir.convert %[[VAL_12]] : (i32) -> ui32
!CHECK: fir.store %[[VAL_13]] to %[[VAL_2]] : !fir.ref<ui32>
!CHECK: %[[VAL_14:.*]] = fir.load %[[VAL_2]] : !fir.ref<ui32>
!CHECK: return %[[VAL_14]] : ui32

subroutine su128(u)
  unsigned(16), intent(in) :: u
  select case (u)
  case (1180591620717411303424u_16)
    continue
  case (170141183460469231731687303715884105728u_16)
    continue
  case (170141183460469231731687303715884105729u_16:340282366920938463463374607431768211454u_16)
    continue
  case (340282366920938463463374607431768211455u_16:)
    continue
  end select
end

!CHECK-LABEL: func.func @_QPsu128(
!CHECK: fir.select_case {{.*}} : ui128
!CHECK-SAME: #fir.point, %c1180591620717411303424_i128{{[_0-9]*}},
!CHECK-SAME: #fir.point, %c-170141183460469231731687303715884105728_i128{{[_0-9]*}},
!CHECK-SAME: #fir.interval, %c-170141183460469231731687303715884105727_i128{{[_0-9]*}}, %c-2_i128{{[_0-9]*}},
!CHECK-SAME: #fir.lower, %c-1_i128{{[_0-9]*}},
