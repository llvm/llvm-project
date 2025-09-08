! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPuse_dsecnds(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<f64>
function use_dsecnds(refTime) result(elapsed)
  double precision :: refTime, elapsed
  elapsed = dsecnds(refTime)
end function

! Verify filename and line operands are passed into runtime call
! CHECK: %[[STRADDR:.*]] = fir.address_of(
! CHECK: %[[LINE:.*]] = arith.constant {{.*}} : i32
! CHECK: %[[FNAME8:.*]] = fir.convert %[[STRADDR]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>

! Call the runtime DSECNDS with (refTime, file, line)
! CHECK: %[[CALL:.*]] = fir.call @_FortranADsecnds(%[[arg0]], %[[FNAME8]], %[[LINE]]) {{.*}} : (!fir.ref<f64>, !fir.ref<i8>, i32) -> f64

! Guard: no illegal ref conversion
! CHECK-NOT: fir.convert {{.*}} : (f64) -> !fir.ref<f64>

! Function returns f64
! CHECK: return {{.*}} : f64

