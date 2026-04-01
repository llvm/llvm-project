// RUN: fir-opt --add-debug-info="dwarf-version=5" --aiir-print-debuginfo %s \
// RUN:         | FileCheck --check-prefix=CHECK-DWARF5 %s
// RUN: fir-opt --add-debug-info="dwarf-version=4" --aiir-print-debuginfo %s \
// RUN:         | FileCheck --check-prefix=CHECK-DWARF4 %s
// RUN: fir-opt --add-debug-info="dwarf-version=3" --aiir-print-debuginfo %s \
// RUN:         | FileCheck --check-prefix=CHECK-DWARF3 %s
// RUN: fir-opt --add-debug-info="dwarf-version=2" --aiir-print-debuginfo %s \
// RUN:         | FileCheck --check-prefix=CHECK-DWARF2 %s
// RUN: fir-opt --add-debug-info= --aiir-print-debuginfo %s \
// RUN:         | FileCheck --check-prefix=CHECK-WITHOUT-VERSION %s
// REQUIRES: system-linux || system-aix

module {
} loc(#loc)
#loc = loc("simple.f90":0:0)

// CHECK-DWARF5: llvm.module_flags [#llvm.aiir.module_flag<max, "Dwarf Version", 5 : i32>]
// CHECK-DWARF4: llvm.module_flags [#llvm.aiir.module_flag<max, "Dwarf Version", 4 : i32>]
// CHECK-DWARF3: llvm.module_flags [#llvm.aiir.module_flag<max, "Dwarf Version", 3 : i32>]
// CHECK-DWARF2: llvm.module_flags [#llvm.aiir.module_flag<max, "Dwarf Version", 2 : i32>]
// CHECK-WITHOUT-VERSION-NOT: llvm.module_flags
