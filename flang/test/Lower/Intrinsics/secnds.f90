! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPuse_secnds(
! CHECK-SAME: %arg0: !fir.ref<f32>
function use_secnds(refTime) result(elapsed)
  real :: refTime, elapsed
  elapsed = secnds(refTime)
end function

! File/line operands (don’t match the actual path/number)
! CHECK: %[[STRADDR:.*]] = fir.address_of(
! CHECK: %[[LINE:.*]] = arith.constant {{.*}} : i32
! CHECK: %[[FNAME8:.*]] = fir.convert %[[STRADDR]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>

! Important: pass refTime by address and return a value f32
! CHECK: %[[CALL:.*]] = fir.call @{{.*}}Secnds(%arg0, %[[FNAME8]], %[[LINE]]) {{.*}} : (!fir.ref<f32>, !fir.ref<i8>, i32) -> f32

! Guard against illegal value ->ref conversion of result
! CHECK-NOT: fir.convert {{.*}} : (f32) -> !fir.ref<f32>

! Function returns an f32 value
! CHECK: return {{.*}} : f32

