## Check that a local symbol in an outlined basic block is updated to its
## address in the cold fragment.

# REQUIRES: system-linux, asserts

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   -save-temp-labels %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -nostdlib -Wl,-q,--discard-none
# RUN: link_fdata %s %t.exe %t.fdata
# RUN: llvm-bolt %t.exe -o %t.bolt --lite=0 --data %t.fdata \
# RUN:   --reorder-blocks=ext-tsp --split-functions --split-all-cold \
# RUN:   --print-cfg --print-only=_start > %t.out 2>&1
# RUN: FileCheck %s --check-prefix=CFG < %t.out
# RUN: llvm-nm -n %t.bolt | FileCheck %s --check-prefix=SYMBOL
# RUN: llvm-objdump -d --disassemble-symbols=.L0 %t.bolt \
# RUN:   | FileCheck %s --check-prefix=CODE
# RUN: llvm-bolt %t.exe -o %t.relax.bolt --lite=0 --data %t.fdata \
# RUN:   --reorder-blocks=ext-tsp --split-functions --split-all-cold --relax-exp
# RUN: llvm-nm -n %t.relax.bolt | FileCheck %s --check-prefix=SYMBOL
# RUN: llvm-objdump -d --disassemble-symbols=.L0 %t.relax.bolt \
# RUN:   | FileCheck %s --check-prefix=CODE

# CFG: IsMultiEntry: 0
# SYMBOL: t .L0
# CODE: <.L0>:
# CODE-NEXT: {{.*}} add x2, x2, #0x1

  .text
  .globl _start
  .type _start, %function
_start:
# FDATA: 1 _start 0 1 _start 4 0 10
.entry_start:
  cbz x0, 1f
  add x0, x0, #1
  ret
## Numeric labels are assembler-only and do not create retained ELF symbols.
1:
  sub x0, x0, #1
  nop
.L0:
  add x2, x2, #1
  ret
  .size _start, .-_start

  .reloc 0, R_AARCH64_NONE
