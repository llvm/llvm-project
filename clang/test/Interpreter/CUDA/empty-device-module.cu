// Tests host-only inputs. They produce an empty device module, and emitting
// PTX for it must not be reported as a failure just because no pass changed
// the module.
// RUN: cat %s | clang-repl --cuda | FileCheck %s

extern "C" int printf(const char*, ...);

int host_only = 42;
printf("host_only: %d\n", host_only);
// CHECK: host_only: 42

__global__ void kernel() {}

kernel<<<1,1>>>();
printf("CUDA Error: %d\n", cudaGetLastError());
// CHECK-NEXT: CUDA Error: 0

%quit
