// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

float process(float *a) [[clang::nonblocking]] { return *a; }
int spinlock(int *a) [[clang::blocking]] { return *a; }

// Without the -fsanitize=realtime flag, we shouldn't attach the attributes.
// CHECK-NOT: {{.*sanitize_realtime .*}}
// CHECK-NOT: {{.*sanitize_realtime_blocking .*}}
