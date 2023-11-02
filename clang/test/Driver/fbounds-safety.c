// RUN: %clang -c %s -### 2>&1 | FileCheck -check-prefix T0 %s
// T0-NOT: -fexperimental-bounds-safety

// RUN: %clang -fexperimental-bounds-safety -### %s 2>&1 | FileCheck -check-prefix T1 %s
// T1: -fexperimental-bounds-safety

// RUN: %clang -fexperimental-bounds-safety -fno-experimental-bounds-safety -c %s -### 2>&1 | FileCheck -check-prefix T2 %s
// T2-NOT: -fexperimental-bounds-safety

// RUN: %clang -fno-experimental-bounds-safety -fexperimental-bounds-safety -c %s -### 2>&1 | FileCheck -check-prefix T3 %s
// T3: -fexperimental-bounds-safety