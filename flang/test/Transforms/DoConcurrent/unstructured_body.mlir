// Tests mapping of a `do concurrent` loop whose body holds raw branching (here
// a SELECT CASE), folded into an scf.execute_region. The region moves into the
// OpenMP loop nest unchanged.

// RUN: fir-opt --omp-do-concurrent-conversion="map-to=host" %s | FileCheck %s --check-prefixes=CHECK,HOST
// RUN: fir-opt --omp-do-concurrent-conversion="map-to=device" %s | FileCheck %s --check-prefixes=CHECK,DEVICE

// CHECK-LABEL: func.func @select_case
func.func @select_case(%arg0: !fir.box<!fir.array<?xf32>>, %arg1: !fir.ref<i32>, %arg2: !fir.ref<i32>) {
  %0:2 = hlfir.declare %arg0 uniq_name("_QFselect_caseEa") : (!fir.box<!fir.array<?xf32>>) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
  %1:2 = hlfir.declare %arg1 uniq_name("_QFselect_caseEn") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %2:2 = hlfir.declare %arg2 uniq_name("_QFselect_caseEtag") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  %c1_i32 = arith.constant 1 : i32
  %3 = fir.convert %c1_i32 : (i32) -> index
  %4 = fir.load %1#0 : !fir.ref<i32>
  %5 = fir.convert %4 : (i32) -> index
  %c1 = arith.constant 1 : index

  // HOST:          %[[TAG:.*]]:2 = hlfir.declare %{{.*}} uniq_name("_QFselect_caseEtag")
  // HOST:          omp.parallel {
  // HOST:            omp.wsloop {

  // The tag is only used inside the region, and is still mapped to the device.
  // DEVICE:        %[[TAG_MAP:.*]] = omp.map.info {{.*}} name("_QFselect_caseEtag")
  // DEVICE:        omp.target {{.*}} map_entries({{.*}}%[[TAG_MAP]] -> %[[TAG_ARG:[^,]*]],
  // DEVICE:          %[[TAG:.*]]:2 = hlfir.declare %[[TAG_ARG]] uniq_name("_QFselect_caseEtag")
  // DEVICE:          omp.teams {
  // DEVICE:            omp.parallel {
  // DEVICE:              omp.distribute {
  // DEVICE:                omp.wsloop {

  // CHECK:                 omp.loop_nest (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
  // CHECK:                   fir.store
  // CHECK-NEXT:              scf.execute_region no_inline {
  // CHECK-NEXT:                %[[TAG_VAL:.*]] = fir.load %[[TAG]]#0 : !fir.ref<i32>
  // CHECK-NEXT:                %[[C1:.*]] = arith.constant 1 : i32
  // CHECK-NEXT:                fir.select_case %[[TAG_VAL]] : i32 [#fir.point, %[[C1]], ^[[CASE1:.*]], unit, ^[[DEFAULT:.*]]]
  // CHECK:                   ^[[CASE1]]:
  // CHECK:                     hlfir.assign
  // CHECK:                     cf.br ^[[EXIT:.*]]
  // CHECK:                   ^[[DEFAULT]]:
  // CHECK:                     hlfir.assign
  // CHECK:                     cf.br ^[[EXIT]]
  // CHECK:                   ^[[EXIT]]:
  // CHECK-NEXT:                scf.yield
  // CHECK-NEXT:              }
  // CHECK-NEXT:              omp.yield
  // CHECK-NEXT:            }

  // CHECK-NOT: fir.do_concurrent
  fir.do_concurrent {
    %6 = fir.alloca i32 {bindc_name = "i"}
    %7:2 = hlfir.declare %6 uniq_name("_QFselect_caseEi") : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    fir.do_concurrent.loop (%arg3) = (%3) to (%5) step (%c1) {
      %8 = fir.convert %arg3 : (index) -> i32
      fir.store %8 to %7#0 : !fir.ref<i32>
      scf.execute_region no_inline {
        %9 = fir.load %2#0 : !fir.ref<i32>
        %c1_i32_0 = arith.constant 1 : i32
        fir.select_case %9 : i32 [#fir.point, %c1_i32_0, ^bb1, unit, ^bb2]
      ^bb1:
        %cst = arith.constant 1.000000e+00 : f32
        %10 = fir.load %7#0 : !fir.ref<i32>
        %11 = fir.convert %10 : (i32) -> i64
        %12 = hlfir.designate %0#0 (%11) : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
        hlfir.assign %cst to %12 : f32, !fir.ref<f32>
        cf.br ^bb3
      ^bb2:
        %cst_1 = arith.constant 0.000000e+00 : f32
        %13 = fir.load %7#0 : !fir.ref<i32>
        %14 = fir.convert %13 : (i32) -> i64
        %15 = hlfir.designate %0#0 (%14) : (!fir.box<!fir.array<?xf32>>, i64) -> !fir.ref<f32>
        hlfir.assign %cst_1 to %15 : f32, !fir.ref<f32>
        cf.br ^bb3
      ^bb3:
        scf.yield
      }
    }
  }

  return
}
