// Test the name table that the compile unit asks for.

// RUN: fir-opt --add-debug-info --mlir-print-debuginfo %s \
// RUN:   | FileCheck %s --check-prefix=NONE
// RUN: fir-opt --add-debug-info="name-table-kind=None" --mlir-print-debuginfo \
// RUN:   %s | FileCheck %s --check-prefix=NONE
// RUN: fir-opt --add-debug-info="name-table-kind=Default" \
// RUN:   --mlir-print-debuginfo %s | FileCheck %s --check-prefix=DEFAULT
// RUN: fir-opt --add-debug-info="name-table-kind=GNU" --mlir-print-debuginfo \
// RUN:   %s | FileCheck %s --check-prefix=GNU
// RUN: fir-opt --add-debug-info="name-table-kind=Apple" --mlir-print-debuginfo \
// RUN:   %s | FileCheck %s --check-prefix=APPLE

// A unit that only emits line directives gets no name table, even when one
// is asked for.
// RUN: fir-opt --add-debug-info="debug-level=DebugDirectivesOnly name-table-kind=Default" \
// RUN:   --mlir-print-debuginfo %s | FileCheck %s --check-prefix=NONE

module {
  func.func @_QPsb() {
    return loc(#loc_sb)
  } loc(#loc_sb)
} loc(#loc_module)
#loc_module = loc("./simple.f90":1:1)
#loc_sb = loc("./simple.f90":2:1)

// NONE: #llvm.di_compile_unit<{{.*}}nameTableKind = None
// DEFAULT: #llvm.di_compile_unit<
// DEFAULT-NOT: nameTableKind
// GNU: #llvm.di_compile_unit<{{.*}}nameTableKind = GNU
// APPLE: #llvm.di_compile_unit<{{.*}}nameTableKind = Apple
