# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo '.tbss; .globl b; b:' | llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o
# RUN: ld.lld -shared -soname=t1.so %t1.o -o %t1.so

# RUN: ld.lld -shared %t.o %t1.o -o %t.so
# RUN: llvm-readobj -r %t.so | FileCheck --check-prefix=GD-REL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck --check-prefix=GD %s

# RUN: ld.lld %t.o %t1.o -o %t
# RUN: llvm-readelf -r %t | FileCheck --check-prefix=NOREL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=LE %s

# RUN: ld.lld %t.o %t1.so -o %t
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=IE-REL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=IE %s

# GD-REL:      .rela.dyn {
# GD-REL-NEXT:   0x2380 R_X86_64_TLSDESC a 0x0
# GD-REL-NEXT:   0x2390 R_X86_64_TLSDESC b 0x0
# GD-REL-NEXT: }

# 0x2380-0x12cf = 4273
# GD:            leaq 4273(%rip), %rax
# GD-NEXT: 12cf: callq *(%rax)
# GD-NEXT:       movl %fs:(%rax), %eax

# 0x2390-0x12db = 4277
# GD-NEXT:       leaq 4277(%rip), %rax
# GD-NEXT: 12db: callq *(%rax)
# GD-NEXT:       movl %fs:(%rax), %eax

# NOREL: no relocations

## offset(a) = -4
# LE:      movq $-4, %rax
# LE-NEXT: nop
# LE-NEXT: movl %fs:(%rax), %eax
## offset(b) = 0
# LE:      movq $0, %rax
# LE-NEXT: nop
# LE-NEXT: movl %fs:(%rax), %eax

# IE-REL:      .rela.dyn {
# IE-REL-NEXT:   0x202360 R_X86_64_TPOFF64 b 0x0
# IE-REL-NEXT: }

## a is relaxed to use LE.
# IE:              movq $-4, %rax
# IE-NEXT:         nop
# IE-NEXT:         movl %fs:(%rax), %eax
## 0x202360 - 0x20129b = 4293
# IE-NEXT:         movq 4293(%rip), %rax
# IE-NEXT: 20129b: nop
# IE-NEXT:         movl %fs:(%rax), %eax

leaq a@tlsdesc(%rip), %rax
call *a@tlscall(%rax)
movl %fs:(%rax), %eax

leaq b@tlsdesc(%rip), %rax
call *b@tlscall(%rax)
movl %fs:(%rax), %eax

.section .tbss
.globl a
.zero 8
a:
.zero 4
