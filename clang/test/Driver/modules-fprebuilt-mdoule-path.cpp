// RUN: %clang -std=c++2a -fprebuilt-module-path=. -### -c %s 2>&1 | FileCheck %s
// RUN: %clang -std=c++20 -fprebuilt-module-path=. -### -c %s 2>&1 | FileCheck %s
// RUN: %clang -std=c++23 -fprebuilt-module-path=. -### -c %s 2>&1 | FileCheck %s
// RUN: %clang -std=c++latest -fprebuilt-module-path=. -### -c %s 2>&1 | FileCheck %s
//
// CHECK-NOT: warning: argument unused during compilation
// CHECK: -fprebuilt-module-path=.
