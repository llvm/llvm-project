## Calls and address pairs in retained functions follow moved targets.
# RUN: llvm-mc -triple=riscv64 -filetype=obj %s -o %t.o
# RUN: ld.lld --emit-relocs --no-relax -Ttext=0x10000 -e _start %t.o -o %t.exe
# RUN: llvm-bolt %t.exe --skip-funcs=source,_start --force-patch -o %t.bolt
# RUN: llvm-objdump -d %t.bolt | FileCheck %s
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
# CHECK-LABEL: <source>:
# CHECK-NEXT: jal t1, {{.*}} <middle>
# CHECK-LABEL: 0000000000400000 <leaf>:

## JITLink reports pairs outside the HI20/LO12 range.
# FAR: BOLT-ERROR: JITLink failed: {{.*}}out of range

  .option norvc
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
