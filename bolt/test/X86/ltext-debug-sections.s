## Test that BOLT correctly handles binaries with .ltext (SHF_X86_64_LARGE)
## sections. Functions from .ltext should be emitted back to .ltext with
## the SHF_X86_64_LARGE flag preserved.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe --emit-relocs -e _start
# RUN: llvm-bolt %t.exe -o %t.bolt --update-debug-sections --reorder-blocks=none \
# RUN:   2>&1 | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-readelf -S %t.bolt | FileCheck %s --check-prefix=CHECK-SECTIONS
# RUN: llvm-objdump --syms %t.bolt | FileCheck %s --check-prefix=CHECK-SYMS

## Verify BOLT detects large code model.
# CHECK-BOLT: large code model detected (.ltext section found)

## Verify original sections are renamed and new sections preserve flags.
# CHECK-SECTIONS: .bolt.org.ltext {{.*}} AXl
# CHECK-SECTIONS: .bolt.org.text  {{.*}} AX
# CHECK-SECTIONS: .ltext          {{.*}} AXl
# CHECK-SECTIONS: .text           {{.*}} AX

## Verify large_func lands in .ltext and _start lands in .text.
# CHECK-SYMS-DAG: .ltext{{.*}} large_func
# CHECK-SYMS-DAG: .text{{.*}} _start

  .text
  .globl _start
  .type _start, @function
_start:
  call large_func
  xorl %eax, %eax
  retq
  .size _start, .-_start

## Large code model function in .ltext with SHF_X86_64_LARGE flag.
  .section .ltext,"axl",@progbits
  .globl large_func
  .type large_func, @function
large_func:
  pushq %rbp
  movq %rsp, %rbp
  movl $42, %eax
  popq %rbp
  retq
  .size large_func, .-large_func
