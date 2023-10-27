! RUN: bbc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: c.func @_QPis_finite_test
subroutine is_finite_test(x, y)
  use ieee_arithmetic, only: ieee_is_finite
  real(4) x
  real(8) y

  ! CHECK:     %[[V_3:[0-9]+]] = fir.load %arg0 : !fir.ref<f32>
  ! CHECK:     %[[V_4:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_3]]) <{bit = 504 : i32}> : (f32) -> i1
  ! CHECK:     %[[V_5:[0-9]+]] = fir.convert %[[V_4]] : (i1) -> !fir.logical<4>
  ! CHECK:     %[[V_6:[0-9]+]] = fir.convert %[[V_5]] : (!fir.logical<4>) -> i1
  ! CHECK:     %[[V_7:[0-9]+]] = fir.call @_FortranAioOutputLogical(%{{.*}}, %[[V_6]]) {{.*}} : (!fir.ref<i8>, i1) -> i1
  print*, ieee_is_finite(x)

  ! CHECK:     %[[V_12:[0-9]+]] = fir.load %arg0 : !fir.ref<f32>
  ! CHECK:     %[[V_13:[0-9]+]] = fir.load %arg0 : !fir.ref<f32>
  ! CHECK:     %[[V_14:[0-9]+]] = arith.addf %[[V_12]], %[[V_13]] {{.*}} : f32
  ! CHECK:     %[[V_15:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_14]]) <{bit = 504 : i32}> : (f32) -> i1
  ! CHECK:     %[[V_16:[0-9]+]] = fir.convert %[[V_15]] : (i1) -> !fir.logical<4>
  ! CHECK:     %[[V_17:[0-9]+]] = fir.convert %[[V_16]] : (!fir.logical<4>) -> i1
  ! CHECK:     %[[V_18:[0-9]+]] = fir.call @_FortranAioOutputLogical(%{{.*}}, %[[V_17]]) {{.*}} : (!fir.ref<i8>, i1) -> i1
  print*, ieee_is_finite(x+x)

  ! CHECK:     %[[V_23:[0-9]+]] = fir.load %arg1 : !fir.ref<f64>
  ! CHECK:     %[[V_24:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_23]]) <{bit = 504 : i32}> : (f64) -> i1
  ! CHECK:     %[[V_25:[0-9]+]] = fir.convert %[[V_24]] : (i1) -> !fir.logical<4>
  ! CHECK:     %[[V_26:[0-9]+]] = fir.convert %[[V_25]] : (!fir.logical<4>) -> i1
  ! CHECK:     %[[V_27:[0-9]+]] = fir.call @_FortranAioOutputLogical(%{{.*}}, %[[V_26]]) {{.*}} : (!fir.ref<i8>, i1) -> i1
  print*, ieee_is_finite(y)

  ! CHECK:     %[[V_32:[0-9]+]] = fir.load %arg1 : !fir.ref<f64>
  ! CHECK:     %[[V_33:[0-9]+]] = fir.load %arg1 : !fir.ref<f64>
  ! CHECK:     %[[V_34:[0-9]+]] = arith.addf %[[V_32]], %[[V_33]] {{.*}} : f64
  ! CHECK:     %[[V_35:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_34]]) <{bit = 504 : i32}> : (f64) -> i1
  ! CHECK:     %[[V_36:[0-9]+]] = fir.convert %[[V_35]] : (i1) -> !fir.logical<4>
  ! CHECK:     %[[V_37:[0-9]+]] = fir.convert %[[V_36]] : (!fir.logical<4>) -> i1
  ! CHECK:     %[[V_38:[0-9]+]] = fir.call @_FortranAioOutputLogical(%{{.*}}, %[[V_37]]) {{.*}} : (!fir.ref<i8>, i1) -> i1
  print*, ieee_is_finite(y+y)
end subroutine is_finite_test

! CHECK-LABEL: c.func @_QQmain
  real(4) x
  real(8) y
  ! CHECK:     %[[V_0:[0-9]+]] = fir.alloca f64 {adapt.valuebyref}
  ! CHECK:     %[[V_1:[0-9]+]] = fir.alloca f32 {adapt.valuebyref}
  ! CHECK:     %cst = arith.constant 3.40282347E+38 : f32
  ! CHECK:     fir.store %cst to %[[V_1]] : !fir.ref<f32>
  ! CHECK:     %cst_0 = arith.constant 1.7976931348623157E+308 : f64
  ! CHECK:     fir.store %cst_0 to %[[V_0]] : !fir.ref<f64>
  ! CHECK:     fir.call @_QPis_finite_test(%[[V_1]], %[[V_0]]) {{.*}} : (!fir.ref<f32>, !fir.ref<f64>) -> ()
  call is_finite_test(huge(x), huge(y))
end
