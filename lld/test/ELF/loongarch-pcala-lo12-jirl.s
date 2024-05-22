# REQUIRES: loongarch

# RUN: llvm-mc --filetype=obj --triple=loongarch32-unknown-elf %s -o %t.la32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64-unknown-elf %s -o %t.la64.o

# RUN: ld.lld %t.la32.o -o %t.la32
# RUN: ld.lld %t.la64.o -o %t.la64
# RUN: llvm-objdump -d --no-show-raw-insn %t.la32 | FileCheck %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.la64 | FileCheck %s
# CHECK:      pcalau12i $t0, -1
# CHECK-NEXT: jirl  $ra, $t0, 564
# CHECK-NEXT: pcalau12i $t0, 0
# CHECK-NEXT: jirl  $zero, $t0, -1348

## PLT shouldn't get generated in this case.
# CHECK-NOT:  Disassembly of section .plt:

.p2align 12
.org 0x234
.global foo
foo:
    li.w    $a0, 42
    ret

.org 0xabc
.global bar
bar:
    li.w    $a7, 94
    syscall 0

.org 0x1000
.global _start
_start:
## The nops are for pushing the relocs off page boundary, to better see the
## page-aligned semantics in action.
    nop
    nop
    nop
    pcalau12i   $t0, %pc_hi20(foo)
    jirl        $ra, $t0, %pc_lo12(foo)
    pcalau12i   $t0, %pc_hi20(bar)
    jirl        $zero, $t0, %pc_lo12(bar)
