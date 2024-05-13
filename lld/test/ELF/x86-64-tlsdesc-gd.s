# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo '.tbss; .globl c; c: .zero 4' | llvm-mc -filetype=obj -triple=x86_64 - -o %t1.o
# RUN: ld.lld -shared -soname=t1.so %t1.o -o %t1.so

# RUN: ld.lld -shared %t.o %t1.o -o %t.so
# RUN: llvm-readobj -r -x .got %t.so | FileCheck --check-prefix=GD-RELA %s
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t.so | FileCheck --check-prefix=GD %s

# RUN: ld.lld -shared %t.o %t1.o -o %t-rel.so -z rel
# RUN: llvm-readobj -r -x .got %t-rel.so | FileCheck --check-prefix=GD-REL %s

# RUN: ld.lld %t.o %t1.o -o %t
# RUN: llvm-readelf -r %t | FileCheck --check-prefix=NOREL %s
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck --check-prefix=LE %s

# RUN: ld.lld %t.o %t1.so -o %t
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=IE-REL %s
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck --check-prefix=IE %s

# GD-RELA:      .rela.dyn {
# GD-RELA-NEXT:   0x23D0 R_X86_64_TLSDESC - 0xB
# GD-RELA-NEXT:   0x23B0 R_X86_64_TLSDESC a 0x0
# GD-RELA-NEXT:   0x23C0 R_X86_64_TLSDESC c 0x0
# GD-RELA-NEXT: }
# GD-RELA:      Hex dump of section '.got':
# GD-RELA-NEXT: 0x000023b0 00000000 00000000 00000000 00000000
# GD-RELA-NEXT: 0x000023c0 00000000 00000000 00000000 00000000
# GD-RELA-NEXT: 0x000023d0 00000000 00000000 00000000 00000000

# GD-REL:       .rel.dyn {
# GD-REL-NEXT:    0x23B8 R_X86_64_TLSDESC -
# GD-REL-NEXT:    0x2398 R_X86_64_TLSDESC a
# GD-REL-NEXT:    0x23A8 R_X86_64_TLSDESC c
# GD-REL-NEXT:  }
# GD-REL:       Hex dump of section '.got':
# GD-REL-NEXT:  0x00002398 00000000 00000000 00000000 00000000
# GD-REL-NEXT:  0x000023a8 00000000 00000000 00000000 00000000
# GD-REL-NEXT:  0x000023b8 00000000 00000000 0b000000 00000000

## &.rela.dyn[a]-pc = 0x23B0-0x12e7 = 4297
# GD:            leaq 4297(%rip), %rax
# GD-NEXT: 12e7: callq *(%rax)
# GD-NEXT:       movl %fs:(%rax), %eax

## &.rela.dyn[b]-pc = 0x23D0-0x12f3 = 4317
# GD-NEXT:       leaq 4317(%rip), %rcx
# GD-NEXT: 12f3: movq %rcx, %rax
# GD-NEXT:       callq *(%rax)
# GD-NEXT:       movl %fs:(%rax), %eax

## &.rela.dyn[c]-pc = 0x23C0-0x1302 = 4286
# GD-NEXT:       leaq 4286(%rip), %r15
# GD-NEXT: 1302: movq %r15, %rax
# GD-NEXT:       callq *(%rax)
# GD-NEXT:       movl %fs:(%rax), %eax

# NOREL: no relocations

## tpoff(a) = st_value(a) - tls_size = -8
# LE:      movq $-8, %rax
# LE-NEXT: nop
# LE-NEXT: movl %fs:(%rax), %eax
## tpoff(b) = st_value(b) - tls_size = -5
# LE:      movq $-5, %rcx
# LE-NEXT: movq %rcx, %rax
# LE-NEXT: nop
# LE-NEXT: movl %fs:(%rax), %eax
## tpoff(c) = st_value(c) - tls_size = -4
# LE:      movq $-4, %r15
# LE-NEXT: movq %r15, %rax
# LE-NEXT: nop
# LE-NEXT: movl %fs:(%rax), %eax

# IE-REL:      .rela.dyn {
# IE-REL-NEXT:   0x202378 R_X86_64_TPOFF64 c 0x0
# IE-REL-NEXT: }

## a is relaxed to use LE.
# IE:              movq $-4, %rax
# IE-NEXT:         nop
# IE-NEXT:         movl %fs:(%rax), %eax
# IE-NEXT:         movq $-1, %rcx
# IE-NEXT:         movq %rcx, %rax
# IE-NEXT:         nop
# IE-NEXT:         movl %fs:(%rax), %eax
## &.rela.dyn[c]-pc = 0x202378 - 0x2012aa = 4302
# IE-NEXT:         movq 4302(%rip), %r15
# IE-NEXT: 2012aa: movq %r15, %rax
# IE-NEXT:         nop
# IE-NEXT:         movl %fs:(%rax), %eax

leaq a@tlsdesc(%rip), %rax
call *a@tlscall(%rax)
movl %fs:(%rax), %eax

## leaq/call may not be adjacent:  https://gitlab.freedesktop.org/mesa/mesa/-/issues/5665
## Test non-RAX registers as well.
leaq b@tlsdesc(%rip), %rcx
movq %rcx, %rax
call *b@tlscall(%rax)
movl %fs:(%rax), %eax

leaq c@tlsdesc(%rip), %r15
movq %r15, %rax
call *c@tlscall(%rax)
movl %fs:(%rax), %eax

.section .tbss
.globl a
.zero 8
a:
.zero 3
b:
.zero 1
