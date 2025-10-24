// RUN: fir-opt --omp-do-concurrent-conversion="map-to=device" %s -o - | FileCheck %s

fir.local {type = local} @_QFfooEmy_local_private_f32 : f32

func.func @_QPfoo() {
  %0 = fir.dummy_scope : !fir.dscope
  %3 = fir.alloca f32 {bindc_name = "my_local", uniq_name = "_QFfooEmy_local"}
  %4:2 = hlfir.declare %3 {uniq_name = "_QFfooEmy_local"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)

  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index

  fir.do_concurrent {
    %7 = fir.alloca i32 {bindc_name = "i"}
    %8:2 = hlfir.declare %7 {uniq_name = "_QFfooEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

    fir.do_concurrent.loop (%arg0) = (%c1) to (%c10) step (%c1) local(@_QFfooEmy_local_private_f32 %4#0 -> %arg1 : !fir.ref<f32>) {
      %9 = fir.convert %arg0 : (index) -> i32
      fir.store %9 to %8#0 : !fir.ref<i32>
      %10:2 = hlfir.declare %arg1 {uniq_name = "_QFfooEmy_local"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
      %cst = arith.constant 4.200000e+01 : f32
      hlfir.assign %cst to %10#0 : f32, !fir.ref<f32>
    }
  }
  return
}

// CHECK: omp.private {type = private} @[[OMP_PRIVATIZER:.*.omp]] : f32

// CHECK: %[[LOCAL_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "{{.*}}my_local"}
// CHECK: %[[LOCAL_MAP:.*]] = omp.map.info var_ptr(%[[LOCAL_DECL]]#1 : {{.*}})

// CHECK: omp.target host_eval({{.*}}) map_entries({{.*}}, %[[LOCAL_MAP]] -> %[[LOCAL_MAP_ARG:.*]] : {{.*}}) {
// CHECK:   %[[LOCAL_DEV_DECL:.*]]:2 = hlfir.declare %[[LOCAL_MAP_ARG]] {uniq_name = "_QFfooEmy_local"}

// CHECK:   omp.teams {
// CHECK:     omp.parallel private(@[[OMP_PRIVATIZER]] %[[LOCAL_DEV_DECL]]#0 -> %[[LOCAL_PRIV_ARG:.*]] : {{.*}}) {
// CHECK:       omp.distribute {
// CHECK:         omp.wsloop {
// CHECK:           omp.loop_nest {{.*}} {
// CHECK:             %[[LOCAL_LOOP_DECL:.*]]:2 = hlfir.declare %[[LOCAL_PRIV_ARG]] {uniq_name = "_QFfooEmy_local"}
// CHECK:             hlfir.assign %{{.*}} to %[[LOCAL_LOOP_DECL]]#0
// CHECK:             omp.yield
// CHECK:           }
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:   }
// CHECK: }
