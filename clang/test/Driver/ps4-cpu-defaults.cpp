// Check that on the PS4 we default to:
// -target-cpu btver2, no exceptions and not -tune-cpu generic

// RUN: %clang -target x86_64-scei-ps4 -c %s -### 2>&1 | FileCheck %s
// CHECK: "-target-cpu" "btver2"
// CHECK-NOT: exceptions
// CHECK-NOT: "-tune-cpu"
