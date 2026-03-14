# This test checks instrumentation of static binary on AArch64.

# REQUIRES: system-linux,bolt-runtime,target=aarch64{{.*}}

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -static
# RUN: llvm-bolt -instrument -instrumentation-sleep-time=1 %t.exe \
# RUN:  -o %t.instr 2>&1 | FileCheck %s
# RUN: llvm-objdump --disassemble-symbols=__bolt_fini_trampoline %t.instr -D \
# RUN:  | FileCheck %s -check-prefix=CHECK-ASM

# CHECK: BOLT-INFO: output linked against instrumentation runtime library
# CHECK-ASM: <__bolt_fini_trampoline>:
# CHECK-ASM-NEXT: ret

  .text
  .align 4
  .global _start
  .type _start, %function
_start:
   bl foo
   ret
  .size _start, .-_start

  .global foo
  .type foo, %function
foo:
  mov	w0, wzr
  ret
  .size foo, .-foo
