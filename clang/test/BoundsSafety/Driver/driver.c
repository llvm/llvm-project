// RUN: %clang -c %s -### 2>&1 | FileCheck -check-prefix T0 %s
// T0-NOT: -fbounds-safety-experimental

// RUN: %clang -fbounds-safety-experimental -### %s 2>&1 | FileCheck -check-prefix T1 %s
// T1: -fbounds-safety-experimental

// RUN: %clang -fbounds-safety-experimental -fno-bounds-safety-experimental -c %s -### 2>&1 | FileCheck -check-prefix T2 %s
// T2-NOT: -fbounds-safety-experimental

// RUN: %clang -fno-bounds-safety-experimental -fbounds-safety-experimental -c %s -### 2>&1 | FileCheck -check-prefix T3 %s
// T3: -fbounds-safety-experimental