// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

float process(float *a) [[clang::nonblocking]] { return *a; }

// Without the -fsanitize=realtime flag, we shouldn't attach the attribute.
// CHECK-NOT: {{.*sanitize_realtime.*}}
