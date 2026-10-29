// Verify runtime doesn't contain compiler-emitted memcpy/memmove calls.
//
// REQUIRES: shared_unwind, x86_64-target-arch

// RUN: %if !asan-dynamic-runtime %{ %clang_asan -O1 %s -o %t %}
// RUN: llvm-objdump -d -l %if asan-dynamic-runtime %{ %shared_libasan %} %else %{ %t %} | FileCheck --implicit-check-not="{{(callq|jmpq) .*<(__interceptor_.*)?mem(cpy|set|move)>}}" %s

int main() { return 0; }
