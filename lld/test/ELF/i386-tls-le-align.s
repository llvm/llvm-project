# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=i386 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t | FileCheck %s

# -alignTo(p_memsz, p_align) = -alignTo(4, 64) = -64

# CHECK: movl %gs:0xffffffc0, %eax

  movl %gs:a@NTPOFF, %eax

.section .tbss,"awT"
.align 64
a:
.long 0
.size a, 4
