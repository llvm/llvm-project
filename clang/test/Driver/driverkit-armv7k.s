// RUN: %clang -c -x assembler-with-cpp -target armv7k-apple-driverkit21.0 -### %s 2>&1 | FileCheck %s
// CHECK: -cc1as
// CHECK-SAME: "-target-cpu" "cortex-a7"
.foo:
vfms.f64 d1, d0, d3