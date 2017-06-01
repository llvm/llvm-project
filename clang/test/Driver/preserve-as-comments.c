// RUN: %clang -S -fno-preserve-as-comments %s -### 2>&1 | FileCheck %s
// CHECK: "-fno-preserve-as-comments"
