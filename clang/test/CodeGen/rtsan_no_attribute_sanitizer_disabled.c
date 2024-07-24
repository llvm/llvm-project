// RUN: %clang     -target x86_64-unknown-linux %s -S -emit-llvm -o - | FileCheck %s


float process(float *a) [[clang::nonblocking]] { return *a; }

// Without the -fsanitize=realtime flag, we shouldn't attach
// the attribute
// CHECK-NOT: {{.*sanitize_realtime.*}}
