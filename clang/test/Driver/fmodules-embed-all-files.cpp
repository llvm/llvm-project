// RUN: %clang -std=c++20 %s -fmodules-embed-all-files -### 2>&1 | FileCheck %s
// CHECK: -fmodules-embed-all-files
