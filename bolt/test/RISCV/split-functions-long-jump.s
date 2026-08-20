## Check that a branch crossing a split-function fragment is redirected through
## a local trampoline. The trampoline uses AUIPC+JALR instead of JAL so the
## cold fragment can be placed outside the +/-1 MiB JAL range.

# RUN: llvm-mc -triple riscv64 -mattr=+c -filetype=obj -o %t.o %s
# RUN: ld.lld --emit-relocs -e _start -o %t.exe %t.o
# RUN: llvm-bolt %t.exe -o %t.bolt -split-functions \
# RUN:   -split-strategy=random2 -bolt-seed=1
# RUN: llvm-objdump -d %t.bolt | FileCheck %s
# RUN: llvm-readelf -s %t.bolt | FileCheck --check-prefix=SYMBOLS %s
# RUN: llvm-mc -triple riscv32 -mattr=+c -filetype=obj -o %t.32.o %s
# RUN: ld.lld --emit-relocs -e _start -o %t.32.exe %t.32.o
# RUN: llvm-bolt %t.32.exe -o %t.32.bolt -split-functions \
# RUN:   -split-strategy=random2 -bolt-seed=1
# RUN: llvm-objdump -d %t.32.bolt | FileCheck %s
# RUN: llvm-readelf -s %t.32.bolt | FileCheck --check-prefix=SYMBOLS %s
# RUN: llvm-mc -triple riscv32 -mattr=+e -filetype=obj -o %t.e.o %s
# RUN: ld.lld --emit-relocs -e _start -o %t.e.exe %t.e.o
# RUN: llvm-bolt %t.e.exe -o %t.e.bolt -split-functions \
# RUN:   -split-strategy=random2 -bolt-seed=1
# RUN: llvm-objdump -d %t.e.bolt 2>&1 | FileCheck %s
# RUN: llvm-readelf -s %t.e.bolt | FileCheck --check-prefix=SYMBOLS %s

# CHECK: Disassembly of section .text:
# CHECK-LABEL: <_start>:
# CHECK: auipc [[REG:[a-z0-9]+]],
# CHECK-NEXT: {{(jalr zero,|jr)}} {{.*}}([[REG]])
# CHECK: Disassembly of section .text.cold:
# CHECK-LABEL: <secondary>:
# SYMBOLS: FUNC GLOBAL DEFAULT {{[0-9]+}} secondary
# SYMBOLS-NOT: $x{{.*}}.cold

  .text
  .globl _start
  .type _start, @function
_start:
  beq a0, zero, .Lcold
  .globl secondary
  .type secondary, @function
secondary:
  li a0, 1
  ret
.Lcold:
  li a0, 2
  ret
  .size _start, .-_start

  .reloc 0, R_RISCV_NONE
