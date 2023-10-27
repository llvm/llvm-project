// Verify runtime doesn't contain compiler-emitted memcpy/memmove calls.
//
// REQUIRES: shared_unwind, x86_64-target-arch

// RUN: %clang_msan -O1 %s -o %t
// RUN: llvm-objdump -d -l %t | FileCheck --implicit-check-not="{{(callq|jmpq) .*<(__interceptor_.*)?mem(cpy|set|move)>}}" %s

int main() { return 0; }
