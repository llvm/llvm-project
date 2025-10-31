! RUN: bbc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPuse_dsecnds(
! CHECK-SAME: %[[arg0:.*]]: !fir.ref<f64>
function use_dsecnds(refTime) result(elapsed)
  double precision :: refTime, elapsed
  elapsed = dsecnds(refTime)
end function

! The argument is lowered with hlfir.declare, which returns two results.
! Capture it here to check that the correct SSA value (%...#0)
! is passed to the runtime call later
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[arg0]] dummy_scope

! The file name and source line are also lowered and passed as runtime arguments
! Capture the constant line number and convert the file name to i8*.
! CHECK: %[[STRADDR:.*]] = fir.address_of(
! CHECK: %[[LINE:.*]] = arith.constant {{.*}} : i32
! CHECK: %[[FNAME8:.*]] = fir.convert %[[STRADDR]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>

! Verify the runtime call is made with:
!   - the declared refTime value (%[[DECL]]#0)
!   - the converted filename
!   - the source line constant
! CHECK: %[[CALL:.*]] = fir.call @_FortranADsecnds(%[[DECL]]#0, %[[FNAME8]], %[[LINE]]) {{.*}} : (!fir.ref<f64>, !fir.ref<i8>, i32) -> f64

! Ensure there is no illegal conversion of a value result into a reference
! CHECK-NOT: fir.convert {{.*}} : (f64) -> !fir.ref<f64>

! Confirm the function result is returned as a plain f64
! CHECK: return {{.*}} : f64


