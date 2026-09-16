// RUN: %clang --target=x86_64-linux-gnu -fsanitize=undefined -fsanitize-trap-loop %s -### 2>&1 | FileCheck %s
// CHECK: "-fsanitize-trap-loop"

