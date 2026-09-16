// Test the dwarf-debug-flags option will show up in the generated debug info.
// RUN: fir-opt --add-debug-info="debug-level=Full dwarf-debug-flags=\"-grecord-command-line -O2\"" \
// RUN:   --mlir-print-debuginfo %s | FileCheck --check-prefix=CHECK-FLAGS %s

module {
  func.func @_QPs() {
    return loc(#loc_s)
  } loc(#loc_s)
} loc(#loc_module)
#loc_module = loc("simple.f90":1:1)
#loc_s = loc("simple.f90":2:1)

// CHECK-FLAGS: #llvm.di_compile_unit<
// CHECK-FLAGS: producer = "{{.*}} -grecord-command-line -O2"
