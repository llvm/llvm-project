! RUN: bbc -emit-fir -o - %s | FileCheck %s

! CHECK-LABEL: c.func @_QPout
subroutine out(x)
  use ieee_arithmetic
  integer, parameter :: k = 8

  ! CHECK:     %[[V_59:[0-9]+]] = fir.alloca f64 <{bindc_name = "r", uniq_name = "_QFoutEr"}>
  ! CHECK:     %[[V_60:[0-9]+]] = fir.declare %[[V_59]] uniq_name("_QFoutEr") : (!fir.ref<f64>) -> !fir.ref<f64>
  ! CHECK:     %[[V_61:[0-9]+]] = fir.declare %arg0 dummy_scope %{{[0-9]+}} arg {{[0-9]+}} uniq_name("_QFoutEx") : (!fir.ref<f64>, !fir.dscope) -> !fir.ref<f64>
  real(k) :: x, r

  ! CHECK:     %[[V_62:[0-9]+]] = fir.load %[[V_61]] : !fir.ref<f64>
  ! CHECK:     %[[V_63:[0-9]+]] = arith.bitcast %[[V_62]] : f64 to i64
  ! CHECK:     %[[V_64:[0-9]+]] = arith.cmpf oeq, %[[V_62]], %cst{{[_0-9]*}} {{.*}} : f64
  ! CHECK:     %[[V_65:[0-9]+]] = fir.if %[[V_64]] -> (f64) {
  ! CHECK:       %[[V_66:[0-9]+]] = fir.call @_FortranAMapException(%c4{{.*}}) fastmath<contract> : (i32) -> i32
  ! CHECK:       fir.call {{.*}}feraiseexcept(%[[V_66]]) fastmath<contract> : (i32)
  ! CHECK:       fir.result %cst{{[_0-9]*}} : f64
  ! CHECK:     } else {
  ! CHECK:       %[[V_66:[0-9]+]] = arith.shli %[[V_63]], %c1{{.*}} : i64
  ! CHECK:       %[[V_67:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_62]]) <{bit = 504 : i32}> : (f64) -> i1
  ! CHECK:       %[[V_68:[0-9]+]] = fir.if %[[V_67]] -> (f64) {
  ! CHECK:         %[[V_69:[0-9]+]] = "llvm.intr.is.fpclass"(%[[V_62]]) <{bit = 360 : i32}> : (f64) -> i1
  ! CHECK:         %[[V_70:[0-9]+]] = fir.if %[[V_69]] -> (f64) {
  ! CHECK:           %[[V_71:[0-9]+]] = arith.shrui %[[V_66]], %c53{{.*}} : i64
  ! CHECK:           %[[V_72:[0-9]+]] = arith.subi %[[V_71]], %c1023{{.*}} : i64
  ! CHECK:           %[[V_73:[0-9]+]] = fir.convert %[[V_72]] : (i64) -> f64
  ! CHECK:           fir.result %[[V_73]] : f64
  ! CHECK:         } else {
  ! CHECK:           %[[V_71:[0-9]+]] = arith.shli %[[V_63]], %c12{{.*}} : i64
  ! CHECK:           %[[V_72:[0-9]+]] = math.ctlz %[[V_71]] : i64
  ! CHECK:           %[[V_73:[0-9]+]] = fir.convert %[[V_72]] : (i64) -> i32
  ! CHECK:           %[[V_74:[0-9]+]] = arith.subi %c-1023{{.*}}, %[[V_73]] : i32
  ! CHECK:           %[[V_75:[0-9]+]] = fir.convert %[[V_74]] : (i32) -> f64
  ! CHECK:           fir.result %[[V_75]] : f64
  ! CHECK:         }
  ! CHECK:         fir.result %[[V_70]] : f64
  ! CHECK:       } else {
  ! CHECK:         %[[V_69:[0-9]+]] = arith.shrui %[[V_66]], %c1{{.*}} : i64
  ! CHECK:         %[[V_70:[0-9]+]] = arith.bitcast %[[V_69]] : i64 to f64
  ! CHECK:         fir.result %[[V_70]] : f64
  ! CHECK:       }
  ! CHECK:       fir.result %[[V_68]] : f64
  ! CHECK:     }
  ! CHECK:     fir.store %[[V_65]] to %[[V_60]] : !fir.ref<f64>
  r = ieee_logb(x)
end
