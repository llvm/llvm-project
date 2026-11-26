## Calls and address pairs in retained functions follow moved targets.
## GOT loads keep addressing their original slot, which is updated separately.
# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs --no-relax --gc-sections -Ttext=0x10000 -e _start %t.o -o %t.exe
# RUN: llvm-bolt %t.exe --skip-funcs=source,_start --force-patch -o %t.bolt
# RUN: llvm-objdump -d %t.bolt | FileCheck %s
# RUN: llvm-objdump -s --section=.got %t.bolt | FileCheck %s --check-prefix=GOT
# RUN: not llvm-bolt %t.exe --skip-funcs=source,_start --custom-allocation-vma=0x90000000 -o %t.far 2>&1 | FileCheck %s --check-prefix=FAR

## The eight-byte leaf entry fits a tail trampoline that preserves t0.
# CHECK-LABEL: <leaf.org.0>:
# CHECK-NEXT: auipc t1,
# CHECK-NEXT: jr {{.*}}t1{{.*}} <leaf>

## Preserve the call pair and patch both PC-relative and absolute addresses.
# CHECK-LABEL: 0000000000010008 <middle>:
# CHECK-NEXT: auipc t0, 0x3f0
# CHECK-NEXT: jalr t0, {{.*}} <leaf>
# CHECK: auipc a1, 0x3f0
# CHECK: addi a1, a1, -0x10
# CHECK-NEXT: lui a2, 0x400
# CHECK-NEXT: mv a2, a2
# CHECK: auipc a3, 0x3f0
# CHECK-NEXT: jalr t0, {{.*}} <leaf>
# CHECK: auipc a0, 0x1
# CHECK-NEXT: ld a0, 0x2c(a0)
# CHECK-LABEL: <source>:
# CHECK-NEXT: jal t1, {{.*}} <middle>
# CHECK-LABEL: 0000000000400000 <leaf>:

## The leaf's GOT slot changes from 0x10000 to 0x400000.
# GOT: 00004000 00000000

## JITLink reports pairs outside the HI20/LO12 range.
# FAR: BOLT-ERROR: JITLink failed: {{.*}}out of range

## patchELFGOT cannot update linker-defined symbols such as _end.
# RUN: ld.lld --emit-relocs --no-relax --gc-sections -Ttext=0x10000 -e end_refs -u leaf %t.o -o %t.end
# RUN: not llvm-bolt %t.end --skip-funcs=end_refs -o %t.end.bolt 2>&1 | FileCheck %s --check-prefix=END
# END: BOLT-ERROR: cannot update RISC-V instruction reference to _end

  .option norvc
  .option norelax
  .text
  .globl leaf, middle, source, _start
  .type leaf, @function
leaf:
  nop
  jr t0
  .size leaf, .-leaf

  .type middle, @function
middle:
  call t0, leaf
.Lhi:
  auipc a1, %pcrel_hi(leaf)
  nop
  addi a1, a1, %pcrel_lo(.Lhi)
  lui a2, %hi(leaf)
  addi a2, a2, %lo(leaf)
.Lindirect:
  auipc a3, %pcrel_hi(leaf)
  jalr t0, %pcrel_lo(.Lindirect)(a3)
.Lgot:
  auipc a0, %got_pcrel_hi(leaf)
  ld a0, %pcrel_lo(.Lgot)(a0)
  ret
  .size middle, .-middle

## Calls with a nonstandard link register retain their target.
  .type source, @function
source:
  jal t1, middle
  ret
  .size source, .-source

  .type _start, @function
_start:
  call source
  ret
  .size _start, .-_start

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
