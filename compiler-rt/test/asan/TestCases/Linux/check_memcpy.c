// Verify runtime doesn't contain compiler-emitted memcpy/memmove calls.
//
// REQUIRES: shared_unwind, x86_64-target-arch
// On Android, %t is the android_run.py wrapper script (the ELF binary is %t.real)
// and ASan is dynamically linked rather than linked into the executable.
// UNSUPPORTED: android

// RUN: %clang_asan -O1 %s -o %t
// RUN: llvm-objdump -d -l %t | FileCheck --implicit-check-not="{{(callq|jmpq) .*<(__interceptor_.*)?mem(cpy|set|move)>}}" %s

int main() { return 0; }
