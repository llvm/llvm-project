// Check that sanitizer prints the faulting instruction bytes on
// dump_instruction_bytes=1

// clang-format off
// RUN: %clangxx  %s -o %t
// RUN: %env_tool_opts=dump_instruction_bytes=1 not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-DUMP
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-NODUMP
// clang-format on

// REQUIRES: x86-target-arch
// FIXME: implement in other sanitizers.
// XFAIL: tsan

int main() {
#if defined(__x86_64__)
  asm("movq $0, %rax");
  asm("movl $0xcafebabe, 0x0(%rax)");
#elif defined(i386)
  asm("movl $0, %eax");
  asm("movl $0xcafebabe, 0x0(%eax)");
#endif
  // CHECK-DUMP: First 16 instruction bytes at pc: c7 00 be ba fe ca
  // CHECK-NODUMP-NOT: First 16 instruction bytes
  return 0;
}
