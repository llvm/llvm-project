! Test lowering of intrinsic subroutines to HLFIR what matters here
! is not to test each subroutine, but to check how their
! lowering interfaces with the rest of lowering.
! RUN: bbc -emit-fir -hlfir -o - %s | FileCheck %s

subroutine test_subroutine(x)
 real :: x
 call cpu_time(x)
end subroutine
! CHECK-LABEL: func.func @_QPtest_subroutine(
! CHECK:  %[[VAL_1:.*]]:2 = hlfir.declare %{{.*}}
! CHECK:  %[[VAL_2:.*]] = fir.call @_FortranACpuTime() fastmath<contract> : () -> f64
! CHECK:  %[[VAL_3:.*]] = fir.convert %[[VAL_2]] : (f64) -> f32
! CHECK:  fir.store %[[VAL_3]] to %[[VAL_1]]#1 : !fir.ref<f32>
