! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s

! CHECK-LABEL: func.func @_QQmain
use ieee_arithmetic
! CHECK:     %[[X:.*]] = fir.alloca f32 {bindc_name = "x", uniq_name = "_QFEx"}
! CHECK:     %[[XDECL:.*]]:2 = hlfir.declare %[[X]] {uniq_name = "_QFEx"}
! CHECK:     %cst = arith.constant -2.000000e+00 : f32
! CHECK:     hlfir.assign %cst to %[[XDECL]]#0 : f32, !fir.ref<f32>
x = -2.0

! CHECK:     %[[V:.*]] = fir.load %[[XDECL]]#0 : !fir.ref<f32>
! CHECK:     %[[BITS:.*]] = arith.bitcast %[[V]] : f32 to i32
! CHECK:     %[[SHIFTED:.*]] = arith.shrui %[[BITS]], %{{.*}} : i32
! CHECK:     %[[LOG:.*]] = fir.convert %[[SHIFTED]] : (i32) -> !fir.logical<4>
! CHECK:     %[[BIT:.*]] = fir.convert %[[LOG]] : (!fir.logical<4>) -> i1
! CHECK:     fir.call @_FortranAioOutputLogical(%{{.*}}, %[[BIT]]) {{.*}} : (!fir.ref<i8>, i1) -> i1

! CHECK:     %cst{{.*}} = arith.constant 1.700000e+01 : f32
! CHECK:     %[[BITS2:.*]] = arith.bitcast %cst{{.*}} : f32 to i32
! CHECK:     %[[SHIFTED2:.*]] = arith.shrui %[[BITS2]], %{{.*}} : i32
! CHECK:     %[[LOG2:.*]] = fir.convert %[[SHIFTED2]] : (i32) -> !fir.logical<4>
! CHECK:     %[[BIT2:.*]] = fir.convert %[[LOG2]] : (!fir.logical<4>) -> i1
! CHECK:     fir.call @_FortranAioOutputLogical(%{{.*}}, %[[BIT2]]) {{.*}} : (!fir.ref<i8>, i1) -> i1
print*, ieee_signbit(x), ieee_signbit(17.0)
end
