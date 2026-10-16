! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPuse_secnds(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<f32>
function use_secnds(refTime) result(elapsed)
  real :: refTime, elapsed
  elapsed = secnds(refTime)
end function

! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]]{{.*}} : (!fir.ref<f32>, !fir.dscope) -> (!fir.ref<f32>, !fir.ref<f32>)

! File/line operands (don't match the actual path/number)
! CHECK: %[[STRADDR:.*]] = fir.address_of(
! CHECK: %[[LINE:.*]] = arith.constant {{.*}} : i32
! CHECK: %[[FNAME8:.*]] = fir.convert %[[STRADDR]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>

! Important: pass refTime by address and return a value f32
! CHECK: %[[CALL:.*]] = fir.call @{{.*}}Secnds(%[[DECL]]#0, %[[FNAME8]], %[[LINE]]) {{.*}} : (!fir.ref<f32>, !fir.ref<i8>, i32) -> f32

! Guard against illegal value ->ref conversion of result
! CHECK-NOT: fir.convert {{.*}} : (f32) -> !fir.ref<f32>

! Function returns an f32 value
! CHECK: return {{.*}} : f32
