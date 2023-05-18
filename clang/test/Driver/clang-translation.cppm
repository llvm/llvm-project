// RUN: %clang %s --precompile -### 2>&1 | FileCheck %s
// CHECK: "-o" "{{[^"]*}}clang-translation.pcm"
