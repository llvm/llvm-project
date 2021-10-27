# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo '.tbss; .globl c; c: .zero 4' | llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o
# RUN: ld.lld -shared -soname=t1.so %t1.o -o %t1.so

# RUN: ld.lld -shared %t.o %t1.o -o %t.so
# RUN: llvm-readobj -r -x .got %t.so | FileCheck --check-prefix=GD-RELA %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck --check-prefix=GD %s

# RUN: ld.lld -shared %t.o %t1.o -o %t-rel.so -z rel
# RUN: llvm-readobj -r -x .got %t-rel.so | FileCheck --check-prefix=GD-REL %s

# RUN: ld.lld %t.o %t1.o -o %t
# RUN: llvm-readelf -r %t | FileCheck --check-prefix=NOREL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=LE %s

# RUN: ld.lld %t.o %t1.so -o %t
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=IE-REL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=IE %s

# GD-RELA:      .rela.dyn {
# GD-RELA-NEXT:   0x23B8 R_X86_64_TLSDESC - 0xB
# GD-RELA-NEXT:   0x23A8 R_X86_64_TLSDESC a 0x0
# GD-RELA-NEXT:   0x23C8 R_X86_64_TLSDESC c 0x0
# GD-RELA-NEXT: }
# GD-RELA:      Hex dump of section '.got':
# GD-RELA-NEXT: 0x000023a8 00000000 00000000 00000000 00000000
# GD-RELA-NEXT: 0x000023b8 00000000 00000000 00000000 00000000
# GD-RELA-NEXT: 0x000023c8 00000000 00000000 00000000 00000000

# GD-REL:       .rel.dyn {
# GD-REL-NEXT:    0x23A0 R_X86_64_TLSDESC -
# GD-REL-NEXT:    0x2390 R_X86_64_TLSDESC a
# GD-REL-NEXT:    0x23B0 R_X86_64_TLSDESC c
# GD-REL-NEXT:  }
# GD-REL:       Hex dump of section '.got':
# GD-REL-NEXT:  0x00002390 00000000 00000000 00000000 00000000
# GD-REL-NEXT:  0x000023a0 00000000 00000000 0b000000 00000000
# GD-REL-NEXT:  0x000023b0 00000000 00000000 00000000 00000000

## &.rela.dyn[a]-pc = 0x23A8-0x12e7 = 4289
# GD:            leaq 4289(%rip), %rax
# GD-NEXT: 12e7: callq *(%rax)
# GD-NEXT:       movl %fs:(%rax), %eax

## &.rela.dyn[b]-pc = 0x23B8-0x12f3 = 4293
# GD-NEXT:       leaq 4293(%rip), %rax
# GD-NEXT: 12f3: callq *(%rax)
# GD-NEXT:       movl %fs:(%rax), %eax

## &.rela.dyn[c]-pc = 0x23C8-0x12f3 = 4297
# GD-NEXT:       leaq 4297(%rip), %rax
# GD-NEXT: 12ff: callq *(%rax)
# GD-NEXT:       movl %fs:(%rax), %eax

# NOREL: no relocations

## tpoff(a) = st_value(a) - tls_size = -8
# LE:      movq $-8, %rax
# LE-NEXT: nop
# LE-NEXT: movl %fs:(%rax), %eax
## tpoff(b) = st_value(b) - tls_size = -5
# LE:      movq $-5, %rax
# LE-NEXT: nop
# LE-NEXT: movl %fs:(%rax), %eax
## tpoff(c) = st_value(c) - tls_size = -4
# LE:      movq $-4, %rax
# LE-NEXT: nop
# LE-NEXT: movl %fs:(%rax), %eax

# IE-REL:      .rela.dyn {
# IE-REL-NEXT:   0x202370 R_X86_64_TPOFF64 c 0x0
# IE-REL-NEXT: }

## a is relaxed to use LE.
# IE:              movq $-4, %rax
# IE-NEXT:         nop
# IE-NEXT:         movl %fs:(%rax), %eax
# IE-NEXT:         movq $-1, %rax
# IE-NEXT:         nop
# IE-NEXT:         movl %fs:(%rax), %eax
## &.rela.dyn[c]-pc = 0x202370 - 0x2012a7 = 4297
# IE-NEXT:         movq 4297(%rip), %rax
# IE-NEXT: 2012a7: nop
# IE-NEXT:         movl %fs:(%rax), %eax

leaq a@tlsdesc(%rip), %rax
call *a@tlscall(%rax)
movl %fs:(%rax), %eax

leaq b@tlsdesc(%rip), %rax
call *b@tlscall(%rax)
movl %fs:(%rax), %eax

leaq c@tlsdesc(%rip), %rax
call *c@tlscall(%rax)
movl %fs:(%rax), %eax

.section .tbss
.globl a
.zero 8
a:
.zero 3
b:
.zero 1
