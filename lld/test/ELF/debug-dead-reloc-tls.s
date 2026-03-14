# REQUIRES: x86
## Test we resolve relocations referencing TLS symbols in .debug_* sections to
## a tombstone value if the referenced TLS symbol is discarded.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld --gc-sections %t.o -o %t
# RUN: llvm-objdump -s %t | FileCheck %s

# CHECK:      Contents of section .debug_info:
# CHECK-NEXT:  0000 00000000 00000000 00000000 00000000
# CHECK-NEXT:  0010 00000000 ffffffff

.globl _start
_start:
  ret

.section .tbss,"awT",@nobits
.globl global
local:
global:
  .quad 0

.section .debug_info
## On ppc64, .quad local@dtprel+0x8000 (st_value 0 is supposed to point to
## 0x8000 bytes past the start of ## the dynamic TLS vector. References usually
## have an addend of 0x8000). MIPS is similar. RISC-V uses 0x800.
  .quad local@dtpoff+0x8000
  .quad global@dtpoff+0x8000

## Many other architectures don't use an offset. GCC x86-64 uses a 32-bit value.
  .long global@dtpoff
  .long -1
