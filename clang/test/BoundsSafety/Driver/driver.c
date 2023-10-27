// RUN: %clang -c %s -### 2>&1 | not grep fbounds-safety-experimental

// RUN: %clang -fbounds-safety-experimental -### %s 2>&1 | FileCheck -check-prefix T0 %s
// T0: -fbounds-safety-experimental

// RUN: %clang -fbounds-safety-experimental -fno-bounds-safety-experimental -c %s -### 2>&1 | not grep -e fbounds-safety-experimental

// RUN: %clang -fno-bounds-safety-experimental -fbounds-safety-experimental -c %s -### 2>&1 | FileCheck -check-prefix T1 %s
// T1: -fbounds-safety-experimental