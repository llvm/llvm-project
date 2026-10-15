## A retained instruction cannot reference a moving non-function symbol until
## instruction relocations can be updated safely. Fail on the first such
## reference instead of falling back to entry patches.

# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs --no-relax %t.o -o %t.exe
# RUN: not llvm-bolt %t.exe --skip-funcs=source -o %t.bolt 2>&1 | FileCheck %s
# RUN: llvm-mc -triple=riscv32 -filetype=obj %s -o %t.32.o
# RUN: ld.lld --emit-relocs --no-relax %t.32.o -o %t.32.exe
# RUN: not llvm-bolt %t.32.exe --skip-funcs=source -o %t.32.bolt 2>&1 | FileCheck %s
# RUN: llvm-mc -triple=riscv64 -defsym=TEXT_END=1 -filetype=obj %s -o %t.text.o
# RUN: ld.lld --emit-relocs --no-relax %t.text.o -o %t.text.exe
# RUN: not llvm-bolt %t.text.exe --skip-funcs=source -o %t.text.bolt 2>&1 | FileCheck %s
# RUN: llvm-bolt %t.text.exe --print-cfg --print-only=source -o %t.moved.bolt | FileCheck %s --check-prefix=CFG

# CHECK: BOLT-ERROR: cannot update RISC-V instruction reference to _end at address 0x{{[0-9a-f]+}} in ignored function source
# CHECK-NOT: patching entries
# CFG: auipc a1, %pcrel_hi(_end)
# CFG: lui a1, %hi(_end)
# CFG: addi a1, a1, %lo(_end)

  .option norvc
  .option norelax
  .text
  .globl _start
  .type _start, @function
_start:
  call source
  li a7, 93
  ecall
  .size _start, .-_start

  .globl source
  .type source, @function
source:
  lla a1, _end
  jal t0, target
  lui a1, %hi(_end)
  addi a1, a1, %lo(_end)
  ret
  .size source, .-source

  .globl target
  .type target, @function
target:
  li a0, 0
  .rept 40
  nop
  .endr
  jr t0
  .size target, .-target

## Also test _end at the end of .text, where it must not be replaced with a
## reference to the last function. Ordinary rewritten code can still use it.
  .ifndef TEXT_END
    .bss
    .space 8
  .endif
