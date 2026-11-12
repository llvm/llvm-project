## Check that llvm-bolt patches functions that are getting ignored after their
## CFG was constructed.

# RUN: %clang %cflags %s -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck %s
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=CHECK-OBJDUMP

  .text

## After the CFG is built, the following function will be marked as ignored
## due to the presence of the internal call.
# CHECK:      BOLT-WARNING: will skip the following function
# CHECK-NEXT: internal_call
	.globl internal_call
  .type internal_call, %function
internal_call:
  .cfi_startproc
# CHECK-OBJDUMP:      <internal_call>:
  call .L1
  jmp .L2
# CHECK-OBJDUMP:      jmp
.L1:
  jmp _start
# CHECK-OBJDUMP:      jmp
# CHECK-OBJDUMP-SAME: <_start>
  ret
.L2:
  jmp _start
# CHECK-OBJDUMP:      jmp
# CHECK-OBJDUMP-SAME: <_start>
  .cfi_endproc
  .size internal_call, .-internal_call

  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
  cmpq  %rdi, 1
  jne  .L0
  movq %rdi, %rax
.L0:
  ret
  .cfi_endproc
  .size _start, .-_start
