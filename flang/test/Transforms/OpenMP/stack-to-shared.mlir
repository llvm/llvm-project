// RUN: fir-opt --split-input-file --omp-stack-to-shared %s | FileCheck %s

module attributes {omp.is_target_device = true} {
  omp.declare_reduction @add_reduction_i32 : i32 init {
  ^bb0(%arg0: i32):
    %c0_i32 = arith.constant 0 : i32
    omp.yield(%c0_i32 : i32)
  } combiner {
  ^bb0(%arg0: i32, %arg1: i32):
    %0 = arith.addi %arg0, %arg1 : i32
    omp.yield(%0 : i32)
  }

  omp.private {type = private} @privatizer_i32 : i32
  omp.private {type = firstprivate} @firstprivatizer_i32 : i32 copy {
  ^bb0(%arg0: i32, %arg1: i32):
    omp.yield(%arg0 : i32)
  }

  // Verify that target device functions are searched for allocas shared across
  // threads of a parallel region.
  //
  // Also ensure that all fir.alloca information is adequately forwarded to the
  // new allocation, that uses of the allocation through hlfir.declare are
  // detected and that only the expected types of uses (parallel reduction and
  // non-private uses inside of a parallel region) are replaced.
  // CHECK-LABEL: func.func @standalone_func
  func.func @standalone_func(%lb: i32, %ub: i32, %step: i32) attributes {omp.declare_target = #omp.declaretarget<device_type = (nohost), capture_clause = (to)>} {
    // CHECK: %[[ALLOC_0:.*]] = omp.alloc_shared_mem i32 {uniq_name = "x"} : !fir.ref<i32>
    %0 = fir.alloca i32 {uniq_name = "x"}
    %c = arith.constant 1 : index
    // CHECK: %[[ALLOC_1:.*]] = omp.alloc_shared_mem !fir.char<1,?>(%[[C:.*]] : index), %[[C]] {bindc_name = "y", uniq_name = "y"} : !fir.ref<!fir.char<1,?>>
    %1 = fir.alloca !fir.char<1,?>(%c : index), %c {bindc_name = "y", uniq_name = "y"}
    // CHECK: %{{.*}}:2 = hlfir.declare %[[ALLOC_1]] typeparams %[[C]] {uniq_name = "y"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    %decl:2 = hlfir.declare %1 typeparams %c {uniq_name = "y"} : (!fir.ref<!fir.char<1,?>>, index) -> (!fir.boxchar<1>, !fir.ref<!fir.char<1,?>>)
    // CHECK: %{{.*}} = fir.alloca i32 {uniq_name = "z"}
    %2 = fir.alloca i32 {uniq_name = "z"}
    // CHECK: %[[ALLOC_2:.*]] = omp.alloc_shared_mem i32 {uniq_name = "a"} : !fir.ref<i32>
    %3 = fir.alloca i32 {uniq_name = "a"}
    // CHECK: %{{.*}} = fir.alloca i32 {uniq_name = "b"}
    %4 = fir.alloca i32 {uniq_name = "b"}
    omp.parallel reduction(@add_reduction_i32 %0 -> %arg0 : !fir.ref<i32>) {
      // CHECK: %{{.*}} = fir.alloca i32 {uniq_name = "c"}
      %5 = fir.alloca i32 {uniq_name = "c"}
      %6:2 = fir.unboxchar %decl#0 : (!fir.boxchar<1>) -> (!fir.ref<!fir.char<1,?>>, index)
      omp.wsloop private(@privatizer_i32 %2 -> %arg1, @firstprivatizer_i32 %3 -> %arg2 : !fir.ref<i32>, !fir.ref<i32>) {
        omp.loop_nest (%arg3) : i32 = (%lb) to (%ub) inclusive step (%step) {
          %7 = fir.load %5 : !fir.ref<i32>
          omp.yield
        }
      }
      omp.terminator
    }
    %5 = fir.load %4 : !fir.ref<i32>
    // CHECK: omp.free_shared_mem %[[ALLOC_0]] : !fir.ref<i32>
    // CHECK-NEXT: omp.free_shared_mem %[[ALLOC_1]] : !fir.ref<!fir.char<1,?>>
    // CHECK-NEXT: omp.free_shared_mem %[[ALLOC_2]] : !fir.ref<i32>
    // CHECK-NEXT: return
    return
  }

  // Verify that generic target regions are searched for allocas shared across
  // threads of a parallel region.
  // CHECK-LABEL: func.func @target_generic
  func.func @target_generic() {
    // CHECK: omp.target
    omp.target {
      %c = arith.constant 0 : i32
      // CHECK: %[[ALLOC_0:.*]] = omp.alloc_shared_mem i32 {uniq_name = "x"} : !fir.ref<i32>
      %0 = fir.alloca i32 {uniq_name = "x"}
      // CHECK: omp.teams
      omp.teams {
        // CHECK: %[[ALLOC_1:.*]] = omp.alloc_shared_mem i32 {uniq_name = "y"} : !fir.ref<i32>
        %1 = fir.alloca i32 {uniq_name = "y"}
        // CHECK: omp.distribute
        omp.distribute {
          omp.loop_nest (%arg0) : i32 = (%c) to (%c) inclusive step (%c) {
            // CHECK: %[[ALLOC_2:.*]] = omp.alloc_shared_mem i32 {uniq_name = "z"} : !fir.ref<i32>
            %2 = fir.alloca i32 {uniq_name = "z"}
            // CHECK: omp.parallel
            omp.parallel {
              %3 = fir.load %0 : !fir.ref<i32>
              %4 = fir.load %1 : !fir.ref<i32>
              %5 = fir.load %2 : !fir.ref<i32>
              // CHECK: omp.terminator
              omp.terminator
            }
            // CHECK: omp.free_shared_mem %[[ALLOC_2]] : !fir.ref<i32>
            // CHECK: omp.yield
            omp.yield
          }
        }
        // CHECK: omp.free_shared_mem %[[ALLOC_1]] : !fir.ref<i32>
        // CHECK: omp.terminator
        omp.terminator
      }
      // CHECK: omp.free_shared_mem %[[ALLOC_0]] : !fir.ref<i32>
      // CHECK: omp.terminator
      omp.terminator
    }
    // CHECK: return
    return
  }

  // Make sure that uses not shared across threads on a parallel region inside
  // of target are not incorrectly detected as such if there's another parallel
  // region in the host wrapping the whole target region.
  // CHECK-LABEL: func.func @target_generic_in_parallel
  func.func @target_generic_in_parallel() {
    // CHECK-NOT: omp.alloc_shared_mem
    // CHECK-NOT: omp.free_shared_mem
    omp.parallel {
      omp.target {
        %c = arith.constant 0 : i32
        %0 = fir.alloca i32 {uniq_name = "x"}
        omp.teams {
          %1 = fir.alloca i32 {uniq_name = "y"}
          omp.distribute {
            omp.loop_nest (%arg0) : i32 = (%c) to (%c) inclusive step (%c) {
              %3 = fir.load %0 : !fir.ref<i32>
              %4 = fir.load %1 : !fir.ref<i32>
              omp.parallel {
                omp.terminator
              }
              omp.yield
            }
          }
          omp.terminator
        }
        omp.terminator
      }
      omp.terminator
    }
    // CHECK: return
    return
  }

  // Ensure that allocations within SPMD target regions are not replaced with
  // device shared memory regardless of use.
  // CHECK-LABEL: func.func @target_spmd
  func.func @target_spmd() {
    // CHECK-NOT: omp.alloc_shared_mem
    // CHECK-NOT: omp.free_shared_mem
    omp.target {
      %c = arith.constant 0 : i32
      %0 = fir.alloca i32 {uniq_name = "x"}
      omp.teams {
        %1 = fir.alloca i32 {uniq_name = "y"}
        omp.parallel {
          %2 = fir.alloca i32 {uniq_name = "z"}
          %3 = fir.load %0 : !fir.ref<i32>
          %4 = fir.load %1 : !fir.ref<i32>
          omp.distribute {
            omp.wsloop {
              omp.loop_nest (%arg0) : i32 = (%c) to (%c) inclusive step (%c) {
                %5 = fir.load %2 : !fir.ref<i32>
                omp.yield
              }
            } {omp.composite}
          } {omp.composite}
          omp.terminator
        } {omp.composite}
        omp.terminator
      }
      omp.terminator
    }
    // CHECK: return
    return
  }
}

// -----

// No transformations must be done when targeting the host device.
// CHECK-LABEL: func.func @host_standalone
func.func @host_standalone() {
  // CHECK-NOT: omp.alloc_shared_mem
  // CHECK-NOT: omp.free_shared_mem
  %0 = fir.alloca i32 {uniq_name = "x"}
  omp.parallel {
    %1 = fir.load %0 : !fir.ref<i32>
    omp.terminator
  }
  // CHECK: return
  return
}

// CHECK-LABEL: func.func @host_target
func.func @host_target() {
  // CHECK-NOT: omp.alloc_shared_mem
  // CHECK-NOT: omp.free_shared_mem
  omp.target {
    %c = arith.constant 0 : i32
    %0 = fir.alloca i32 {uniq_name = "x"}
    omp.teams {
      %1 = fir.alloca i32 {uniq_name = "y"}
      omp.distribute {
        omp.loop_nest (%arg0) : i32 = (%c) to (%c) inclusive step (%c) {
          %2 = fir.alloca i32 {uniq_name = "z"}
          omp.parallel {
            %3 = fir.load %0 : !fir.ref<i32>
            %4 = fir.load %1 : !fir.ref<i32>
            %5 = fir.load %2 : !fir.ref<i32>
            omp.terminator
          }
          omp.yield
        }
      }
      omp.terminator
    }
    omp.terminator
  }
  // CHECK: return
  return
}
