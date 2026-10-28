// Tests LOCAL_INIT with multi-dimensional do concurrent (i=1:n, j=1:m).
// Verifies:
//   1. omp.loop_nest has two induction variables and collapse(2)
//   2. Per-iteration copy is inlined at top of the collapsed loop body

// RUN: fir-opt --omp-do-concurrent-conversion="map-to=host" %s | FileCheck %s

fir.local {type = local_init} @_QFtestEtmp_firstprivate_f32 : f32 copy {
^bb0(%arg0: !fir.ref<f32>, %arg1: !fir.ref<f32>):
  %0 = fir.load %arg0 : !fir.ref<f32>
  fir.store %0 to %arg1 : !fir.ref<f32>
  fir.yield(%arg1 : !fir.ref<f32>)
}

func.func @_QPlocal_init_multidim() {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %c20 = arith.constant 20 : index

  %0 = fir.alloca f32 {bindc_name = "tmp", uniq_name = "_QFtestEtmp"}
  %1:2 = hlfir.declare %0 uniq_name("_QFtestEtmp") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)

  %cst = arith.constant 3.140000e+00 : f32
  fir.store %cst to %1#0 : !fir.ref<f32>

  fir.do_concurrent {
    %2 = fir.alloca i32 {bindc_name = "i"}
    %3:2 = hlfir.declare %2 uniq_name("_QFtestEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %4 = fir.alloca i32 {bindc_name = "j"}
    %5:2 = hlfir.declare %4 uniq_name("_QFtestEj") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

    fir.do_concurrent.loop (%arg0, %arg1) = (%c1, %c1) to (%c10, %c20) step (%c1, %c1)
      local(@_QFtestEtmp_firstprivate_f32 %1#0 -> %arg2 : !fir.ref<f32>) {
      %6 = fir.convert %arg0 : (index) -> i32
      fir.store %6 to %3#0 : !fir.ref<i32>
      %7 = fir.convert %arg1 : (index) -> i32
      fir.store %7 to %5#0 : !fir.ref<i32>
      %8:2 = hlfir.declare %arg2 uniq_name("_QFtestEtmp") : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
      %9 = fir.load %8#0 : !fir.ref<f32>
      %10 = fir.load %3#0 : !fir.ref<i32>
      %11 = fir.convert %10 : (i32) -> f32
      %12 = arith.addf %9, %11 fastmath<contract> : f32
      hlfir.assign %12 to %8#0 : f32, !fir.ref<f32>
    }
  }
  return
}

// Verify: omp.private has type=private with NO copy region
// CHECK:      omp.private {type = private} @[[PRIVATIZER:.*_QFtestEtmp.*\.omp]] : f32

// CHECK-LABEL: func.func @_QPlocal_init_multidim
// CHECK:         %[[TMP:.*]] = fir.alloca f32 <{bindc_name = "tmp", {{.*}>}}
// CHECK:         %[[TMP_DECL:.*]]:2 = hlfir.declare %[[TMP]]

// Verify: omp.parallel + omp.wsloop + omp.loop_nest structure
// CHECK:         omp.parallel {
// CHECK:           omp.wsloop private(@[[PRIVATIZER]] %[[TMP_DECL]]#0 -> %[[PRIV:.*]] : !fir.ref<f32>)

// Verify: omp.loop_nest with two induction variables and collapse(2)
// CHECK:             omp.loop_nest (%[[IV_I:.*]], %[[IV_J:.*]]) : index =
// CHECK-SAME:          (%{{.*}}, %{{.*}}) to (%{{.*}}, %{{.*}}) inclusive step (%{{.*}}, %{{.*}}) collapse(2)

// Verify: per-iteration copy is inlined at top of collapsed loop body
// CHECK:               %[[LOADED:.*]] = fir.load %[[TMP_DECL]]#0 : !fir.ref<f32>
// CHECK-NEXT:          fir.store %[[LOADED]] to %[[PRIV]] : !fir.ref<f32>

// Verify: induction variable stores follow the copy
// CHECK:               fir.convert %[[IV_I]]
// CHECK:               fir.convert %[[IV_J]]

// Verify: user code follows
// CHECK:               hlfir.declare %[[PRIV]]
// CHECK:               omp.yield
