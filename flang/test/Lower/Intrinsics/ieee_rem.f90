! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

! CHECK-LABEL: c.func @_QQmain
  use ieee_arithmetic, only: ieee_rem

  ! CHECK:     %[[V_0:[0-9]+]] = fir.alloca f16 {bindc_name = "x2", uniq_name = "_QFEx2"}
  ! CHECK:     %[[V_1:[0-9]+]]:2 = hlfir.declare %[[V_0]] {uniq_name = "_QFEx2"} : (!fir.ref<f16>) -> (!fir.ref<f16>, !fir.ref<f16>)
  ! CHECK:     %[[V_2:[0-9]+]] = fir.alloca f32 {bindc_name = "x4", uniq_name = "_QFEx4"}
  ! CHECK:     %[[V_3:[0-9]+]]:2 = hlfir.declare %[[V_2]] {uniq_name = "_QFEx4"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  ! CHECK:     %[[V_4:[0-9]+]] = fir.alloca f64 {bindc_name = "x8", uniq_name = "_QFEx8"}
  ! CHECK:     %[[V_5:[0-9]+]]:2 = hlfir.declare %[[V_4]] {uniq_name = "_QFEx8"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  ! CHECK:     %[[V_6:[0-9]+]] = fir.alloca f16 {bindc_name = "y2", uniq_name = "_QFEy2"}
  ! CHECK:     %[[V_7:[0-9]+]]:2 = hlfir.declare %[[V_6]] {uniq_name = "_QFEy2"} : (!fir.ref<f16>) -> (!fir.ref<f16>, !fir.ref<f16>)
  ! CHECK:     %[[V_8:[0-9]+]] = fir.alloca f32 {bindc_name = "y4", uniq_name = "_QFEy4"}
  ! CHECK:     %[[V_9:[0-9]+]]:2 = hlfir.declare %[[V_8]] {uniq_name = "_QFEy4"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
  ! CHECK:     %[[V_10:[0-9]+]] = fir.alloca f64 {bindc_name = "y8", uniq_name = "_QFEy8"}
  ! CHECK:     %[[V_11:[0-9]+]]:2 = hlfir.declare %[[V_10]] {uniq_name = "_QFEy8"} : (!fir.ref<f64>) -> (!fir.ref<f64>, !fir.ref<f64>)
  real(2) :: x2, y2
  real(4) :: x4, y4
  real(8) :: x8, y8

  ! CHECK:     hlfir.assign %cst{{[_0-9]*}} to %[[V_3]]#0 : f32, !fir.ref<f32>
  x4 = 3.3_4
  ! CHECK:     hlfir.assign %cst{{[_0-9]*}} to %[[V_9]]#0 : f32, !fir.ref<f32>
  y4 = -0.0_4
  ! CHECK-DAG: %[[V_12:[0-9]+]] = fir.load %[[V_3]]#0 : !fir.ref<f32>
  ! CHECK-DAG: %[[V_13:[0-9]+]] = fir.load %[[V_9]]#0 : !fir.ref<f32>
  ! CHECK-DAG: %[[V_14:[0-9]+]] = fir.convert %[[V_12]] : (f32) -> f32
  ! CHECK-DAG: %[[V_15:[0-9]+]] = fir.convert %[[V_13]] : (f32) -> f32
  ! CHECK-DAG: %[[V_16:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_15]]) <{bit = 516 : i32}> : (f32) -> i1
  ! CHECK-DAG: %[[V_17:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_14]]) <{bit = 144 : i32}> : (f32) -> i1
  ! CHECK-DAG: %[[V_18:[0-9]+]] = arith.andi %[[V_17]], %[[V_16]] : i1
  ! CHECK-DAG: %[[V_19:[0-9]+]] = fir.call @remainderf(%[[V_14]], %[[V_15]]) fastmath<contract> : (f32, f32) -> f32
  ! CHECK:     fir.if %[[V_18]] {
  ! CHECK:       %[[V_40:[0-9]+]] = fir.call @_FortranAMapException(%c16{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_41:[0-9]+]] = fir.call @feraiseexcept(%[[V_40]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     }
  ! CHECK:     %[[V_20:[0-9]+]] = fir.convert %[[V_19]] : (f32) -> f32
  ! CHECK:     hlfir.assign %[[V_20]] to %[[V_3]]#0 : f32, !fir.ref<f32>
  x4 = ieee_rem(x4, y4)
! print*, x4

  ! CHECK:     hlfir.assign %cst{{[_0-9]*}} to %[[V_1]]#0 : f16, !fir.ref<f16>
  x2 = 3.0_2
  ! CHECK:     hlfir.assign %cst{{[_0-9]*}} to %[[V_11]]#0 : f64, !fir.ref<f64>
  y8 = 2.0_8
  ! CHECK-DAG: %[[V_21:[0-9]+]] = fir.load %[[V_1]]#0 : !fir.ref<f16>
  ! CHECK-DAG: %[[V_22:[0-9]+]] = fir.load %[[V_11]]#0 : !fir.ref<f64>
  ! CHECK-DAG: %[[V_23:[0-9]+]] = fir.convert %[[V_21]] : (f16) -> f64
  ! CHECK-DAG: %[[V_24:[0-9]+]] = fir.convert %[[V_22]] : (f64) -> f64
  ! CHECK-DAG: %[[V_25:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_24]]) <{bit = 516 : i32}> : (f64) -> i1
  ! CHECK-DAG: %[[V_26:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_23]]) <{bit = 144 : i32}> : (f64) -> i1
  ! CHECK-DAG: %[[V_27:[0-9]+]] = arith.andi %[[V_26]], %[[V_25]] : i1
  ! CHECK-DAG: %[[V_28:[0-9]+]] = fir.call @remainder(%[[V_23]], %[[V_24]]) fastmath<contract> : (f64, f64) -> f64
  ! CHECK:     fir.if %[[V_27]] {
  ! CHECK:       %[[V_40:[0-9]+]] = fir.call @_FortranAMapException(%c16{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_41:[0-9]+]] = fir.call @feraiseexcept(%[[V_40]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     }
  ! CHECK:     %[[V_29:[0-9]+]] = fir.convert %[[V_28]] : (f64) -> f64
  ! CHECK:     %[[V_30:[0-9]+]] = fir.convert %[[V_29]] : (f64) -> f16
  ! CHECK:     hlfir.assign %[[V_30]] to %[[V_1]]#0 : f16, !fir.ref<f16>
  x2 = ieee_rem(x2, y8)
! print*, x2

  ! CHECK:     hlfir.assign %cst{{[_0-9]*}} to %[[V_5]]#0 : f64, !fir.ref<f64>
  x8 = huge(x8)
  ! CHECK:     hlfir.assign %cst{{[_0-9]*}} to %[[V_7]]#0 : f16, !fir.ref<f16>
  y2 = tiny(y2)
  ! CHECK-DAG: %[[V_31:[0-9]+]] = fir.load %[[V_5]]#0 : !fir.ref<f64>
  ! CHECK-DAG: %[[V_32:[0-9]+]] = fir.load %[[V_7]]#0 : !fir.ref<f16>
  ! CHECK-DAG: %[[V_33:[0-9]+]] = fir.convert %[[V_31]] : (f64) -> f64
  ! CHECK-DAG: %[[V_34:[0-9]+]] = fir.convert %[[V_32]] : (f16) -> f64
  ! CHECK-DAG: %[[V_35:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_34]]) <{bit = 516 : i32}> : (f64) -> i1
  ! CHECK-DAG: %[[V_36:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_33]]) <{bit = 144 : i32}> : (f64) -> i1
  ! CHECK-DAG: %[[V_37:[0-9]+]] = arith.andi %[[V_36]], %[[V_35]] : i1
  ! CHECK-DAG: %[[V_38:[0-9]+]] = fir.call @remainder(%[[V_33]], %[[V_34]]) fastmath<contract> : (f64, f64) -> f64
  ! CHECK:     fir.if %[[V_37]] {
  ! CHECK:       %[[V_40:[0-9]+]] = fir.call @_FortranAMapException(%c16{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:       %[[V_41:[0-9]+]] = fir.call @feraiseexcept(%[[V_40]]) fastmath<contract> : (i32) -> i32
  ! CHECK:     }
  ! CHECK:     %[[V_39:[0-9]+]] = fir.convert %[[V_38]] : (f64) -> f64
  ! CHECK:     hlfir.assign %[[V_39]] to %[[V_5]]#0 : f64, !fir.ref<f64>
  x8 = ieee_rem(x8, y2)
! print*, x8

end
