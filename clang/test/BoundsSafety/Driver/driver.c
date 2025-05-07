// ALL-NOT: unknown argument

// Warning: careful to not grep some directory in the buid
// RUN: %clang -c %s -### 2>&1 | not grep fbounds-safety

// RUN: %clang -fbounds-safety -### %s 2>&1 | FileCheck -check-prefixes ALL,T0 %s
// T0: -fbounds-safety
// T0: -enable-constraint-elimination

// RUN: %clang -fbounds-safety -fno-bounds-safety -c %s -### 2>&1 | not grep -e fbounds-safety -e enable-constraint-elimination

// RUN: %clang -fno-bounds-safety -fbounds-safety -c %s -### 2>&1 | FileCheck -check-prefixes ALL,T4 %s
// T4: -fbounds-safety
// T4: -enable-constraint-elimination

// RUN: %clang -fbounds-attributes -c %s -### 2>&1 | not grep fbounds-attributes
// RUN: %clang -fbounds-attributes -c %s -### 2>&1 | FileCheck -check-prefixes ALL,T6 %s
// T6: -fbounds-safety
// T6: -enable-constraint-elimination

// RUN: %clang -fbounds-attributes -fno-bounds-attributes -c %s -### 2>&1 | not grep -e fbounds-safety -e enable-constraint-elimination
// RUN: %clang -fbounds-attributes -fno-bounds-attributes -c %s -### 2>&1 | not grep -e fbounds-attributes -e enable-constraint-elimination
// RUN: %clang -fbounds-attributes -fno-bounds-attributes -c %s -### 2>&1 | not grep -e fbounds-safety -e enable-constraint-elimination


// RUN: %clang -fbounds-safety -mllvm -enable-constraint-elimination -### %s 2>&1 | grep enable-constraint-elimination -o | wc -l | FileCheck -check-prefixes ALL,T7 %s
// T7: 1

// RUN: %clang -fbounds-safety -mllvm -enable-constraint-elimination=false -### %s 2>&1 | grep enable-constraint-elimination -o | wc -l | FileCheck -check-prefixes ALL,T8 %s
// T8: 1

// RUN: %clang -fbounds-safety -mllvm -enable-constraint-elimination=false -### %s 2>&1 | FileCheck -check-prefixes ALL,T9 %s
// T9: -enable-constraint-elimination=false

// RUN: %clang -fbounds-safety -mllvm -enable-constraint-elimination -mllvm -enable-constraint-elimination=false -### %s 2>&1 | grep enable-constraint-elimination -o | wc -l | FileCheck -check-prefixes ALL,T10 %s
// T10: 2

// RUN: %clang -c %s -### 2>&1 | not grep fexperimental-bounds-safety-cxx
// RUN: %clang -fbounds-safety -c %s -### 2>&1 | not grep fexperimental-bounds-safety-cxx

// RUN: %clang -fbounds-safety -Xclang -fexperimental-bounds-safety-cxx -### %s 2>&1 | FileCheck -check-prefixes ALL,T11 %s
// T11: -fexperimental-bounds-safety-cxx

// RUN: %clang -c %s -### 2>&1 | not grep fexperimental-bounds-safety-objc
// RUN: %clang -fbounds-safety -c %s -### 2>&1 | not grep fexperimental-bounds-safety-objc

// RUN: %clang -fbounds-safety -Xclang -fexperimental-bounds-safety-objc -### %s 2>&1 | FileCheck -check-prefixes ALL,T12 %s
// T12: -fexperimental-bounds-safety-objc

// RUN: %clang -fbounds-safety -c %s -### 2>&1 | FileCheck -check-prefixes ALL,T13 %s
// T13: -fbounds-safety
// T13: -enable-constraint-elimination

// RUN: %clang -fbounds-safety -fno-bounds-safety-relaxed-system-headers -c %s -### 2>&1 | FileCheck -check-prefixes ALL,T14 %s
// T14: -fno-bounds-safety-relaxed-system-headers
