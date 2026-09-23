! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPis_finite_test(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<f32> {{.*}}, %[[ARG1:.*]]: !fir.ref<f64> {{.*}})
subroutine is_finite_test(x, y)
  use ieee_arithmetic, only: ieee_is_finite
  real(4) x
  real(8) y

  ! CHECK-DAG: %[[X_DECL:.*]]:2 = hlfir.declare %[[ARG0]] {{.*}} {uniq_name = "_QFis_finite_testEx"}
  ! CHECK-DAG: %[[Y_DECL:.*]]:2 = hlfir.declare %[[ARG1]] {{.*}} {uniq_name = "_QFis_finite_testEy"}

  ! CHECK:     %[[X_VAL:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<f32>
  ! CHECK:     %[[IS_FINITE_X:.*]] = "llvm.intr.is.fpclass"(%[[X_VAL]]) <{bit = 504 : i32}> : (f32) -> i1
  ! CHECK:     %[[L4_X:.*]] = fir.convert %[[IS_FINITE_X]] : (i1) -> !fir.logical<4>
  ! CHECK:     %[[I1_X:.*]] = fir.convert %[[L4_X]] : (!fir.logical<4>) -> i1
  ! CHECK:     fir.call @_FortranAioOutputLogical({{.*}}, %[[I1_X]])
  print*, ieee_is_finite(x)

  ! CHECK:     %[[X_VAL1:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<f32>
  ! CHECK:     %[[X_VAL2:.*]] = fir.load %[[X_DECL]]#0 : !fir.ref<f32>
  ! CHECK:     %[[X_ADD:.*]] = arith.addf %[[X_VAL1]], %[[X_VAL2]] {{.*}} : f32
  ! CHECK:     %[[IS_FINITE_XADD:.*]] = "llvm.intr.is.fpclass"(%[[X_ADD]]) <{bit = 504 : i32}> : (f32) -> i1
  ! CHECK:     %[[L4_XADD:.*]] = fir.convert %[[IS_FINITE_XADD]] : (i1) -> !fir.logical<4>
  ! CHECK:     %[[I1_XADD:.*]] = fir.convert %[[L4_XADD]] : (!fir.logical<4>) -> i1
  ! CHECK:     fir.call @_FortranAioOutputLogical({{.*}}, %[[I1_XADD]])
  print*, ieee_is_finite(x+x)

  ! CHECK:     %[[Y_VAL:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<f64>
  ! CHECK:     %[[IS_FINITE_Y:.*]] = "llvm.intr.is.fpclass"(%[[Y_VAL]]) <{bit = 504 : i32}> : (f64) -> i1
  ! CHECK:     %[[L4_Y:.*]] = fir.convert %[[IS_FINITE_Y]] : (i1) -> !fir.logical<4>
  ! CHECK:     %[[I1_Y:.*]] = fir.convert %[[L4_Y]] : (!fir.logical<4>) -> i1
  ! CHECK:     fir.call @_FortranAioOutputLogical({{.*}}, %[[I1_Y]])
  print*, ieee_is_finite(y)

  ! CHECK:     %[[Y_VAL1:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<f64>
  ! CHECK:     %[[Y_VAL2:.*]] = fir.load %[[Y_DECL]]#0 : !fir.ref<f64>
  ! CHECK:     %[[Y_ADD:.*]] = arith.addf %[[Y_VAL1]], %[[Y_VAL2]] {{.*}} : f64
  ! CHECK:     %[[IS_FINITE_YADD:.*]] = "llvm.intr.is.fpclass"(%[[Y_ADD]]) <{bit = 504 : i32}> : (f64) -> i1
  ! CHECK:     %[[L4_YADD:.*]] = fir.convert %[[IS_FINITE_YADD]] : (i1) -> !fir.logical<4>
  ! CHECK:     %[[I1_YADD:.*]] = fir.convert %[[L4_YADD]] : (!fir.logical<4>) -> i1
  ! CHECK:     fir.call @_FortranAioOutputLogical({{.*}}, %[[I1_YADD]])
  print*, ieee_is_finite(y+y)
end subroutine is_finite_test

! CHECK-LABEL: func.func @_QQmain()
  real(4) x
  real(8) y
  ! CHECK:     %[[X_HUGE:.*]] = arith.constant 3.40282347E+38 : f32
  ! CHECK:     %[[Y_HUGE:.*]] = arith.constant 1.7976931348623157E+308 : f64
  ! CHECK:     %[[X_HUGE_ASSOC:.*]]:3 = hlfir.associate %[[X_HUGE]] {adapt.valuebyref} : (f32) -> (!fir.ref<f32>, !fir.ref<f32>, i1)
  ! CHECK:     %[[Y_HUGE_ASSOC:.*]]:3 = hlfir.associate %[[Y_HUGE]] {adapt.valuebyref} : (f64) -> (!fir.ref<f64>, !fir.ref<f64>, i1)
  ! CHECK:     fir.call @_QPis_finite_test(%[[X_HUGE_ASSOC]]#0, %[[Y_HUGE_ASSOC]]#0)
  ! CHECK:     hlfir.end_associate %[[X_HUGE_ASSOC]]#1, %[[X_HUGE_ASSOC]]#2
  ! CHECK:     hlfir.end_associate %[[Y_HUGE_ASSOC]]#1, %[[Y_HUGE_ASSOC]]#2
  call is_finite_test(huge(x), huge(y))
end
