// RUN: touch %t.o
// RUN: %clang --param ssp-buffer-size=1 %t.o -### 2>&1 | FileCheck %s
// CHECK-NOT: warning: argument unused during compilation: '--param ssp-buffer-size=1'
