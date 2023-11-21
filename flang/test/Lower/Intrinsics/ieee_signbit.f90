! RUN: bbc -emit-fir -hlfir=false -o - %s | FileCheck %s

! CHECK-LABEL: c.func @_QQmain
use ieee_arithmetic
! CHECK:     %[[V_0:[0-9]+]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFEx"}
! CHECK:     %cst = arith.constant -2.000000e+00 : f32
! CHECK:     fir.store %cst to %[[V_0]] : !fir.ref<f32>
x = -2.0

! CHECK:     %[[V_4:[0-9]+]] = fir.load %[[V_0]] : !fir.ref<f32>
! CHECK:     %[[V_5:[0-9]+]] = arith.bitcast %[[V_4]] : f32 to i32
! CHECK:     %[[V_6:[0-9]+]] = arith.shrui %[[V_5]], %c31{{.*}} : i32
! CHECK:     %[[V_7:[0-9]+]] = fir.convert %[[V_6]] : (i32) -> !fir.logical<4>
! CHECK:     %[[V_8:[0-9]+]] = fir.convert %[[V_7]] : (!fir.logical<4>) -> i1
! CHECK:     %[[V_9:[0-9]+]] = fir.call @_FortranAioOutputLogical(%{{.*}}, %[[V_8]]) {{.*}} : (!fir.ref<i8>, i1) -> i1

! CHECK:     %cst_0 = arith.constant 1.700000e+01 : f32
! CHECK:     %[[V_10:[0-9]+]] = arith.bitcast %cst_0 : f32 to i32
! CHECK:     %[[V_11:[0-9]+]] = arith.shrui %[[V_10]], %c31{{.*}} : i32
! CHECK:     %[[V_12:[0-9]+]] = fir.convert %[[V_11]] : (i32) -> !fir.logical<4>
! CHECK:     %[[V_13:[0-9]+]] = fir.convert %[[V_12]] : (!fir.logical<4>) -> i1
! CHECK:     %[[V_14:[0-9]+]] = fir.call @_FortranAioOutputLogical(%{{.*}}, %[[V_13]]) {{.*}} : (!fir.ref<i8>, i1) -> i1
print*, ieee_signbit(x), ieee_signbit(17.0)
end
