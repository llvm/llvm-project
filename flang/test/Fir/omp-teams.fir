// RUN: fir-opt --fir-to-llvm-ir %s | FileCheck %s

// Tests that private allocations of an `omp.teams` op remain nested inside the
// owning `omp.teams` op after FIR to LLVM conversion.

func.func @foo() {
  omp.teams {
    %2 = fir.alloca f32 {bindc_name = "priv_val", pinned, uniq_name = "_QFEpriv_val"}
    %3 = fir.alloca i32 {bindc_name = "i", pinned, uniq_name = "_QFEi"}
    %c1_i32 = arith.constant 1 : i32
    %c1000_i32 = arith.constant 1000 : i32
    %c1_i32_0 = arith.constant 1 : i32
    omp.distribute {
      omp.loop_nest (%arg0) : i32 = (%c1_i32) to (%c1000_i32) inclusive step (%c1_i32_0) {
        fir.store %arg0 to %3 : !fir.ref<i32>
        omp.yield
      }
    }
    omp.terminator
  }
  return
}

// CHECK-LABEL:   llvm.func @foo() {
// CHECK:           omp.teams {
// CHECK:             %[[TEAMS_IV_ALLOC:.*]] = llvm.alloca %{{.*}} x i32 {bindc_name = "i", pinned}
// CHECK:             %[[TEAMS_PRIV_ALLOC:.*]] = llvm.alloca %{{.*}} x f32 {bindc_name = "priv_val", pinned}
// CHECK:             omp.distribute {
// CHECK:               omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
// CHECK:                 llvm.store %{{.*}}, %[[TEAMS_IV_ALLOC]] : i32, !llvm.ptr
// CHECK:                 omp.yield
// CHECK:               }
// CHECK:             }
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           llvm.return
// CHECK:         }

