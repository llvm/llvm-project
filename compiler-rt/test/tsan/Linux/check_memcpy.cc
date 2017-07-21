// Test that verifies TSan runtime doesn't contain compiler-emitted
// memcpy/memmove calls. It builds the binary with TSan and passes it to
// check_memcpy.sh script.

// RUN: %clangxx_tsan -O1 %s -o %t
// RUN: llvm-objdump -d %t | FileCheck %s

// REQUIRES: compiler-rt-optimized

int main() {
  return 0;
}

// CHECK-NOT: callq {{.*<(__interceptor_)?mem(cpy|set)>}}
// tail calls:
// CHECK-NOT: jmpq {{.*<(__interceptor_)?mem(cpy|set)>}}

