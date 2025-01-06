/// PS4/PS5 miscellaneous toolchain behavior.

// RUN: %clang -c %s -### -target x86_64-scei-ps4 2>&1 | FileCheck %s
// RUN: %clang -c %s -### -target x86_64-sie-ps5 2>&1 | FileCheck %s
// CHECK-DAG: "-ffunction-sections"
// CHECK-DAG: "-fdata-sections"
// CHECK-DAG: "-fdeclspec"

/// Verify LTO is enabled (no diagnostic).
// RUN: %clang %s -### -target x86_64-scei-ps4 -flto 2>&1 | FileCheck %s --check-prefix=LTO
// RUN: %clang %s -### -target x86_64-sie-ps5 -flto 2>&1 | FileCheck %s --check-prefix=LTO
// LTO-NOT: error:
// LTO-NOT: unable to pass LLVM bit-code

// Verify that the jump table sizes section is enabled.
// RUN: %clang %s -target x86_64-sie-ps5 -### 2>&1 | FileCheck -check-prefix=JUMPTABLESIZES %s
// JUMPTABLESIZES: "-mllvm" "-emit-jump-table-sizes-section"
// JUMPTABLESIZES: "-plugin-opt=-emit-jump-table-sizes-section"
