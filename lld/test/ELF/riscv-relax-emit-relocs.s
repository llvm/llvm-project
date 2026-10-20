# REQUIRES: riscv
## Test that we can handle --emit-relocs while relaxing.

# RUN: rm -rf %t && mkdir %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=riscv32 -mattr=+relax %s -o 32.o -riscv-align-rvc=0
# RUN: ld.lld -Ttext=0x10000 --emit-relocs 32.o -o 32
# RUN: llvm-objdump -dr --no-show-raw-insn -M no-aliases 32 | FileCheck %s

# RUN: llvm-mc -filetype=obj -triple=riscv64 -mattr=+relax %s -o 64.o -riscv-align-rvc=0
# RUN: ld.lld -Ttext=0x10000 --emit-relocs 64.o -o 64
# RUN: llvm-objdump -dr --no-show-raw-insn -M no-aliases 64 | FileCheck %s

## -r should keep original relocations.
# RUN: ld.lld -r 64.o -o 64.r
# RUN: llvm-objdump -dr --no-show-raw-insn -M no-aliases 64.r | FileCheck %s --check-prefix=CHECKR

## --no-relax should keep original relocations.
# RUN: ld.lld --emit-relocs --no-relax 64.o -o 64.norelax
# RUN: llvm-objdump -dr --no-show-raw-insn -M no-aliases 64.norelax | FileCheck %s --check-prefix=CHECKNORELAX

# CHECK:      <_start>:
# CHECK-NEXT:     jal ra, 0x10008 <f>
# CHECK-NEXT:         R_RISCV_JAL f
# CHECK-NEXT:         R_RISCV_RELAX *ABS*
# CHECK-NEXT:     jal ra, 0x10008 <f>
# CHECK-NEXT:         R_RISCV_JAL f
# CHECK-NEXT:         R_RISCV_RELAX *ABS*
# CHECK-EMPTY:
# CHECK-NEXT: <f>:
# CHECK-NEXT:     jalr zero, 0x0(ra)
# CHECK-NEXT:         R_RISCV_ALIGN *ABS*+0x4

# CHECKR:      <_start>:
# CHECKR-NEXT:     auipc ra, 0x0
# CHECKR-NEXT:         R_RISCV_CALL_PLT f
# CHECKR-NEXT:         R_RISCV_RELAX *ABS*
# CHECKR-NEXT:     jalr ra, 0x0(ra)
# CHECKR-NEXT:     auipc ra, 0x0
# CHECKR-NEXT:         R_RISCV_CALL_PLT f
# CHECKR-NEXT:         R_RISCV_RELAX *ABS*
# CHECKR-NEXT:     jalr ra, 0x0(ra)
# CHECKR-NEXT:     addi zero, zero, 0x0
# CHECKR-NEXT:         R_RISCV_ALIGN *ABS*+0x4
# CHECKR-EMPTY:
# CHECKR-NEXT: <f>:
# CHECKR-NEXT:     jalr zero, 0x0(ra)

# CHECKNORELAX:      <_start>:
# CHECKNORELAX-NEXT:     auipc ra, 0x0
# CHECKNORELAX-NEXT:         R_RISCV_CALL_PLT f
# CHECKNORELAX-NEXT:         R_RISCV_RELAX *ABS*
# CHECKNORELAX-NEXT:     jalr ra, 0x10(ra)
# CHECKNORELAX-NEXT:     auipc ra, 0x0
# CHECKNORELAX-NEXT:         R_RISCV_CALL_PLT f
# CHECKNORELAX-NEXT:         R_RISCV_RELAX *ABS*
# CHECKNORELAX-NEXT:     jalr ra, 0x8(ra)
# CHECKNORELAX-EMPTY:
# CHECKNORELAX-NEXT: <f>:
# CHECKNORELAX-NEXT:     jalr zero, 0x0(ra)
# CHECKNORELAX-NEXT:         R_RISCV_ALIGN *ABS*+0x4

.global _start
_start:
  call f
  call f
  .balign 8
f:
  ret
