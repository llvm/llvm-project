// Tests mapping `local` locality specifier to `private` clauses for a simple
// case (not `init` or `copy` regions).

// RUN: fir-opt --omp-do-concurrent-conversion="map-to=host" %s | FileCheck %s

fir.local {type = local} @_QFlocal_spec_translationElocal_var_private_f32 : f32

func.func @_QPlocal_spec_translation() {
  %3 = fir.alloca f32 {bindc_name = "local_var", uniq_name = "_QFlocal_spec_translationElocal_var"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFlocal_spec_translationElocal_var"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)

  %c4_i32 = arith.constant 4 : index
  %c11_i32 = arith.constant 11 : index
  %c1 = arith.constant 1 : index

  fir.do_concurrent {
    %7 = fir.alloca i32 {bindc_name = "i"}
    %8:2 = hlfir.declare %7 {uniq_name = "_QFlocal_spec_translationEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

    fir.do_concurrent.loop (%arg0) = (%c4_i32) to (%c11_i32) step (%c1)
      local(@_QFlocal_spec_translationElocal_var_private_f32 %4#0 -> %arg1 : !fir.ref<f32>) {
      %9 = fir.convert %arg0 : (index) -> i32
      fir.store %9 to %8#0 : !fir.ref<i32>

      %10:2 = hlfir.declare %arg1 {uniq_name = "_QFlocal_spec_translationElocal_var"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
      %cst = arith.constant 4.200000e+01 : f32
      hlfir.assign %cst to %10#0 : f32, !fir.ref<f32>
    }
  }
  return
}

// CHECK: omp.private {type = private} @[[PRIVATIZER:.*local_spec_translationElocal_var.*.omp]] : f32

// CHECK: func.func @_QPlocal_spec_translation
// CHECK:   %[[LOCAL_VAR:.*]] = fir.alloca f32 {bindc_name = "local_var", {{.*}}}
// CHECK:   %[[LOCAL_VAR_DECL:.*]]:2 = hlfir.declare %[[LOCAL_VAR]]
// CHECK:   omp.parallel {
// CHECK:     omp.wsloop private(@[[PRIVATIZER]] %[[LOCAL_VAR_DECL]]#0 -> %[[LOCAL_ARG:.*]] : !fir.ref<f32>) {
// CHECK:       omp.loop_nest {{.*}} {
// CHECK:       %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[LOCAL_ARG]]
// CHECK:       %[[C42:.*]] = arith.constant
// CHECK:       hlfir.assign %[[C42]] to %[[PRIV_DECL]]#0
// CHECK:       omp.yield
// CHECK:     }
// CHECK:   }
// CHECK:   omp.terminator
// CHECK: }
