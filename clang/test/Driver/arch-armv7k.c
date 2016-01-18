// Tests that make sure armv7k is mapped to the correct CPU

// RUN: %clang -target x86_64-apple-macosx10.9 -arch armv7k -c %s -### 2>&1 | FileCheck %s
// CHECK: "-cc1"{{.*}} "-target-cpu" "cortex-a7"
// CHECK-NOT: "-fsjlj-exceptions"
