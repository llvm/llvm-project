## Retained GOT loads keep addressing the original GOT slot. The slot is
## updated separately, so their function target can still move.
# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs --no-relax --gc-sections -Ttext=0x10000 %t.o -o %t.exe
# RUN: llvm-bolt %t.exe --skip-funcs=_start --no-huge-pages --use-gnu-stack --align-text=4 --align-functions=4 --custom-allocation-vma=0x20000 -o %t.bolt
# RUN: llvm-objdump -d --no-show-raw-insn %t.bolt | FileCheck %s --check-prefix=CODE
# RUN: llvm-objdump -s --section=.got %t.bolt | FileCheck %s --check-prefix=GOT

# CODE-LABEL: <_start>:
# CODE: auipc a0, 0x1
# CODE-NEXT: ld a0, 0x1c(a0)
# CODE-LABEL: 0000000000020000 <target>:

## The target's GOT slot changes from 0x10010 to 0x20000.
# GOT: 00000200 00000000

## patchELFGOT updates function pointers, but cannot update linker-defined
## symbols such as _end. Keep rejecting these references in retained code.
# RUN: ld.lld --emit-relocs --no-relax --gc-sections -Ttext=0x10000 -e end_refs -u target %t.o -o %t.end
# RUN: not llvm-bolt %t.end --skip-funcs=end_refs -o %t.end.bolt 2>&1 | FileCheck %s --check-prefix=END
# END: BOLT-ERROR: cannot update RISC-V instruction reference to _end

  .option norvc
  .option norelax
  .text
  .globl _start, target
  .type _start, @function
_start:
  nop
.Lgot:
  auipc a0, %got_pcrel_hi(target)
  ld a0, %pcrel_lo(.Lgot)(a0)
  ret
  .size _start, .-_start

  .type target, @function
target:
  li a0, 7
  ret
  .size target, .-target

  .section .text.end_refs,"ax",@progbits
  .globl end_refs
  .type end_refs, @function
end_refs:
  nop
.Lend_got:
  auipc a0, %got_pcrel_hi(_end)
  ld a0, %pcrel_lo(.Lend_got)(a0)
  ret
  .size end_refs, .-end_refs
