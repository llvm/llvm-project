// RUN: fir-opt --lower-workdistribute %s | FileCheck %s

// A Fortran allocatable descriptor (fir.box) crossing the workdistribute target
// fission must be recomputed inside the isolated target from its mapped
// descriptor, not cached by value via __flang_workdistribute_to/from. Caching
// the box would freeze a host base_addr into the device kernel.

// CHECK-LABEL: func.func @recompute_descriptor(
// CHECK: omp.target_data
// The box must not be cached by value across the split.
// CHECK-NOT: __flang_workdistribute
// It is recomputed inside the device target and indexed there.
// CHECK: fir.load %{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
// CHECK: omp.teams
// CHECK: omp.loop_nest
// CHECK: fir.box_addr

module attributes {llvm.target_triple = "amdgcn-amd-amdhsa", omp.is_gpu = true, omp.is_target_device = true} {
func.func @recompute_descriptor(%arg0: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
  %map = omp.map.info var_ptr(%arg0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.heap<!fir.array<?xf32>>>) map_clauses(tofrom) capture(ByRef) name("x") -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
  omp.target kernel_type(generic) map_entries(%map -> %barg : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c9 = arith.constant 9 : index
    %cst = arith.constant 5.000000e+00 : f32
    // The descriptor load crosses the split - it must be recomputed, not cached.
    %box = fir.load %barg : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
    omp.teams {
      omp.workdistribute {
        fir.do_loop %iv = %c0 to %c9 step %c1 unordered {
          %addr = fir.box_addr %box : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
          %coor = fir.coordinate_of %addr, %iv : (!fir.heap<!fir.array<?xf32>>, index) -> !fir.ref<f32>
          fir.store %cst to %coor : !fir.ref<f32>
        }
        omp.terminator
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}
}
