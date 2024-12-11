

// ALL-NOT: unknown argument

// RUN: %clang -c %s -### 2>&1 | not grep fexperimental-bounds-safety-attributes

// RUN: %clang -fexperimental-bounds-safety-attributes -c %s -### 2>&1 | not grep fbounds-safety

// RUN: %clang -fexperimental-bounds-safety-attributes -c %s -### 2>&1 | FileCheck -check-prefixes ALL,T0 %s
// T0: -fexperimental-bounds-safety-attributes

// RUN: %clang -fbounds-safety -fexperimental-bounds-safety-attributes -c %s -### 2>&1 | FileCheck -check-prefixes ALL,T1 %s
// T1: -fexperimental-bounds-safety-attributes

// RUN: %clang -fexperimental-bounds-safety-attributes -fno-experimental-bounds-safety-attributes -c %s -### 2>&1 | not grep fexperimental-bounds-safety-attributes
