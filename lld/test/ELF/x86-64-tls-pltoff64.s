# REQUIRES: x86
## Test dynamic TLS sequences that call __tls_get_addr indirectly through
## R_X86_64_PLTOFF64 instead of a direct call.

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=x86_64 a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 b.s -o b.o
# RUN: ld.lld -shared b.o -soname=b.so -o b.so

# RUN: ld.lld a.o b.so -o out
# RUN: llvm-readelf -Sr out | FileCheck %s --check-prefix=SEC
# RUN: llvm-objdump -d --no-show-raw-insn --no-print-imm-hex out | FileCheck %s --check-prefix=EXE

# RUN: ld.lld -shared a.o b.so -o out.so
# RUN: llvm-readelf -r out.so | FileCheck %s --check-prefix=SDYN
# RUN: llvm-objdump -d --no-show-raw-insn --no-print-imm-hex out.so | FileCheck %s --check-prefix=SHARED

# SEC:      .got PROGBITS 00000000002023c8
# SEC:      Relocation section '.rela.dyn' {{.*}} contains 1 entries:
# SEC-NEXT: Offset
# SEC-NEXT: 00000000002023c8 {{.*}} R_X86_64_TPOFF64 {{.*}} y + 0

## The TLS block is 15 bytes. x1 is at DTPOFF 7 and TPOFF -8, and x2 at DTPOFF 11
## and TPOFF -4. Each optimized sequence is padded with a nop to the original 22 bytes.
# EXE-LABEL: <_start>:
# EXE:         movq %fs:0, %rax
# EXE-NEXT:    leaq -8(%rax), %rax
# EXE-NEXT:    nopw (%rax,%rax)

## y is preemptible. Its GD sequence is optimized to IE.
# EXE-NEXT:    movq %fs:0, %rax
# EXE-NEXT:    addq [[#]](%rip), %rax # 0x2023c8
# EXE-NEXT:    nopw (%rax,%rax)

# EXE-NEXT:    nopw %cs:(%rax,%rax)
# EXE-NEXT:    movq %fs:0, %rax
# EXE-NEXT:    leaq -8(%rax), %rcx
# EXE-NEXT:    leaq -4(%rax), %rdx

# SDYN:      Relocation section '.rela.dyn' {{.*}} contains 4 entries:
# SDYN:      00000000000024e0 {{.*}} R_X86_64_DTPMOD64 0
# SDYN-NEXT: 00000000000024f0 {{.*}} R_X86_64_DTPMOD64 0
# SDYN-NEXT: 0000000000002500 {{.*}} R_X86_64_DTPMOD64 {{.*}} y + 0
# SDYN-NEXT: 0000000000002508 {{.*}} R_X86_64_DTPOFF64 {{.*}} y + 0
# SDYN:      Relocation section '.rela.plt' {{.*}} contains 1 entries:
# SDYN:      {{.*}} R_X86_64_JUMP_SLOT {{.*}} __tls_get_addr + 0

# SHARED-LABEL: <_start>:
# SHARED:        leaq [[#]](%rip), %rdi # 0x24f0
# SHARED-NEXT:   movabsq $-8496, %rax
# SHARED-NEXT:   addq %rbx, %rax
# SHARED-NEXT:   callq *%rax

# SHARED-NEXT:   leaq [[#]](%rip), %rdi # 0x2500
# SHARED-NEXT:   movabsq $-8496, %rax
# SHARED-NEXT:   addq %rbx, %rax
# SHARED-NEXT:   callq *%rax

# SHARED-NEXT:   leaq [[#]](%rip), %rdi # 0x24e0
# SHARED-NEXT:   movabsq $-8496, %rax
# SHARED-NEXT:   addq %r15, %rax
# SHARED-NEXT:   callq *%rax

## x1 is at DTPOFF 7
# SHARED-NEXT:   leaq 7(%rax), %rcx
# SHARED-NEXT:   leaq 11(%rax), %rdx

#--- a.s
.globl _start
_start:
  leaq x1@tlsgd(%rip), %rdi
  movabsq $__tls_get_addr@PLTOFF, %rax
  addq %rbx, %rax
  callq *%rax

  leaq y@tlsgd(%rip), %rdi
  movabsq $__tls_get_addr@PLTOFF, %rax
  addq %rbx, %rax
  callq *%rax

  leaq x1@tlsld(%rip), %rdi
  movabsq $__tls_get_addr@PLTOFF, %rax
  addq %r15, %rax
  callq *%rax
  leaq x1@dtpoff(%rax), %rcx
  leaq x2@dtpoff(%rax), %rdx

.section .tbss,"awT",@nobits
.globl x1, x2
.hidden x1, x2
.space 7
x1: .zero 4
x2: .zero 4

#--- b.s
.section .tbss,"awT",@nobits
.globl y
y:  .zero 4
