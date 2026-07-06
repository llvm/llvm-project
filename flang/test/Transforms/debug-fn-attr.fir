// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s | FileCheck %s

module {
  func.func @_QPvector_magnitude() attributes {fir.proc_attrs = #fir.proc_attrs<pure>} {
    return
  }
  func.func @_QPsquare() attributes {fir.proc_attrs = #fir.proc_attrs<elemental, pure>} {
    return
  }
  func.func @_QPfactorial() attributes {fir.proc_attrs = #fir.proc_attrs<recursive>} {
    return
  }
}

// CHECK: #di_subprogram = #llvm.di_subprogram<{{.*}}, name = "vector_magnitude",{{.*}}subprogramFlags = "Definition|Pure"
// CHECK: #di_subprogram1 = #llvm.di_subprogram<{{.*}}, name = "square",{{.*}}subprogramFlags = "Definition|Pure|Elemental"
// CHECK: #di_subprogram2 = #llvm.di_subprogram<{{.*}}, name = "factorial",{{.*}}subprogramFlags = "Definition|Recursive"
