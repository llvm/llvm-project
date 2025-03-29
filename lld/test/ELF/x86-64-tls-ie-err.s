# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: not ld.lld %t.o -o /dev/null 2>&1 | FileCheck -DFILE=%t.o %s

# CHECK: error: [[FILE]]:(.text+0x2): invalid prefix with R_X86_64_CODE_4_GOTTPOFF!
# CHECK-NEXT: error: [[FILE]]:(.text+0x8): invalid prefix with R_X86_64_CODE_6_GOTTPOFF!
# CHECK-NEXT: error: [[FILE]]:(.text+0x12): R_X86_64_CODE_4_GOTTPOFF must be used in MOVQ or ADDQ instructions only
# CHECK-NEXT: error: [[FILE]]:(.text+0x1a): R_X86_64_CODE_6_GOTTPOFF must be used in ADDQ instructions with NDD/NF/NDD+NF only

## These negative tests are to check if the invalid prefix and unsupported
## instructions for TLS relocation types with APX instructions are handled as
## errors.

.type tls0,@object
.section .tbss,"awT",@nobits
.globl tls0
.align 4
tls0:
 .long 0
 .size tls0, 4

.text
.globl _start
_start:
  addq 0(%rip), %rax, %r16
  .reloc .-4, R_X86_64_CODE_4_GOTTPOFF, tls0-4

  movq 0(%rip), %r16
  .reloc .-4, R_X86_64_CODE_6_GOTTPOFF, tls0-4

  andq 0(%rip), %r16
  .reloc .-4, R_X86_64_CODE_4_GOTTPOFF, tls0-4

  andq 0(%rip), %rax, %r16
  .reloc .-4, R_X86_64_CODE_6_GOTTPOFF, tls0-4
