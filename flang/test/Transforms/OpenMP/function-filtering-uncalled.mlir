
// RUN: fir-opt --omp-function-filtering %s | FileCheck %s

// Test that uncalled internal procedures containing target regions are removed
// on device side

// CHECK-NOT: func.func @uncalled_internal_proc


module attributes {omp.is_target_device = true} {
  func.func @uncalled_internal_proc() -> ()
      attributes {
        fir.host_symbol = @main_program,
        llvm.linkage = #llvm.linkage<internal>,
        omp.declare_target =
          #omp.declaretarget<device_type = (host), capture_clause = (to)>
      } {
    omp.target {
      omp.terminator
    }
    func.return
  }
}
