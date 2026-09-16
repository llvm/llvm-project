## Check that a branch crossing a split-function fragment is redirected through
## a local trampoline. The trampoline uses AUIPC+JALR instead of JAL so the
## cold fragment can be placed outside the +/-1 MiB JAL range. Force a 2 MiB
## gap so linker relaxation cannot shorten the trampoline back into a JAL.
## Also check that --print-longjmp dumps the RV CFG before linker relaxation.

# RUN: llvm-mc -triple riscv64 -mattr=+c -filetype=obj -o %t.o %s
# RUN: ld.lld --emit-relocs -e _start -o %t.exe %t.o
# RUN: llvm-bolt %t.exe -o %t.bolt --split-functions \
# RUN:   -split-strategy=random2 -bolt-seed=1 --pad-funcs-before=_start:2097152 \
# RUN:   --print-longjmp --print-only=_start | FileCheck %s --check-prefix=PRINT
# RUN: llvm-objdump -d --show-all-symbols %t.bolt | FileCheck %s
# RUN: llvm-readobj --symbols %t.bolt | FileCheck --check-prefix=SYMBOLS %s
# RUN: llvm-bolt %t.exe -o %t.unsplit.bolt --split-functions=false
# RUN: llvm-readelf -S %t.unsplit.bolt | FileCheck --check-prefix=UNSPLIT %s

# UNSPLIT-NOT: .text.cold

# PRINT-LABEL: Binary Function "_start" after riscv-long-jmp {
# PRINT: jump {{.*}}, {{[a-z][a-z0-9]*}}

# CHECK: Disassembly of section .text:
# CHECK-LABEL: <_start>:
# CHECK: auipc [[REG:[a-z0-9]+]],
# CHECK-NEXT: {{(jalr zero,|jr)}} {{.*}}([[REG]])
# CHECK: Disassembly of section .text.cold:
# CHECK-LABEL: <secondary>:
# SYMBOLS:      Name: secondary
# SYMBOLS-NEXT: Value:
# SYMBOLS-NEXT: Size: 0
# SYMBOLS-NEXT: Binding: Global
# SYMBOLS-NEXT: Type: Function
# SYMBOLS-NEXT: Other:
# SYMBOLS-NEXT: Section: .text.cold

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
