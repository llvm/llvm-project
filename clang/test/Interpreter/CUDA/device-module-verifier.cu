// Release builds skip the IR verifier. Turn it on and check that the first
// device module passes it.
// RUN: cat %s | clang-repl --cuda -Xcc -fverify-intermediate-code 2>&1 \
// RUN:   | FileCheck %s

extern "C" int printf(const char*, ...);

__global__ void kernel() {}
printf("kernel: %d\n", 0);
// CHECK-NOT: module flag identifiers must be unique
// CHECK-NOT: Broken module found
// CHECK: kernel: 0

%quit
