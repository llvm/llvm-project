// RUN: %clang_cl -### -- %s 2>&1 | FileCheck %s --check-prefix=PRE-CXX20
// RUN: %clang_cl -std:c++20 -### -- %s 2>&1 | FileCheck %s
// RUN: %clang_cl -std:c++20 -### -fdelayed-template-parsing -- %s 2>&1 | FileCheck %s --check-prefix=CHECK-EXPLICIT

// PRE-CXX20: -fdelayed-template-parsing

// CHECK-NOT: -fdelayed-template-parsing

// CHECK-EXPLICIT: warning: -fdelayed-template-parsing is deprecated after C++20
// CHECK-EXPLICIT: -fdelayed-template-parsing
