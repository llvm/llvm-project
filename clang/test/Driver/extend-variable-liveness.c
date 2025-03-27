// Tests that -fextend-variable-liveness and its aliases are correctly passed
// by the driver.

// RUN: %clang -### -c %s 2>&1 | FileCheck %s --check-prefixes=CHECK,DEFAULT
// RUN: %clang -fextend-variable-liveness=none -### -c %s 2>&1 | FileCheck %s --check-prefixes=CHECK,NONE
// RUN: %clang -fextend-variable-liveness=this -### -c %s 2>&1 | FileCheck %s --check-prefixes=CHECK,THIS
// RUN: %clang -fextend-variable-liveness=all -### -c %s 2>&1 | FileCheck %s --check-prefixes=CHECK,ALL
// RUN: %clang -fextend-variable-liveness -### -c %s 2>&1 | FileCheck %s --check-prefixes=CHECK,ALL

// CHECK:       "-cc1"
// DEFAULT-NOT: -fextend-variable-liveness
// NONE-SAME:   "-fextend-variable-liveness=none"
// THIS-SAME:   "-fextend-variable-liveness=this"
// ALL-SAME:    "-fextend-variable-liveness=all"
