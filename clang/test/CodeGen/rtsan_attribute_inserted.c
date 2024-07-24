// RUN: %clang     -target x86_64-unknown-linux -fsanitize=realtime %s -S -emit-llvm -o - | FileCheck %s

float process(float *a) [[clang::nonblocking]] { return *a; }

// CHECK-LABEL: @process{{.*}}#0 {
// CHECK: attributes #0 = {
// CHECK-SAME: {{.*sanitize_realtime.*}}
