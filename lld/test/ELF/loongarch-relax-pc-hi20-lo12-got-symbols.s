## This test verifies that the pair pcalau12i + ld.w/d is relaxed/not relaxed
## depending on the target symbol properties.

# REQUIRES: loongarch
# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc --filetype=obj --triple=loongarch32 -mattr=+32s,+relax symbols.s -o symbols.32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 -mattr=+relax symbols.s -o symbols.64.o
# RUN: llvm-mc --filetype=obj --triple=loongarch32 -mattr=+32s,+relax abs.s -o abs.32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 -mattr=+relax abs.s -o abs.64.o

# RUN: ld.lld --shared -Tlinker.t symbols.32.o abs.32.o -o symbols.32.so
# RUN: ld.lld --shared -Tlinker.t symbols.64.o abs.64.o -o symbols.64.so
# RUN: llvm-objdump -d --no-show-raw-insn symbols.32.so | FileCheck --check-prefixes=LIB %s
# RUN: llvm-objdump -d --no-show-raw-insn symbols.64.so | FileCheck --check-prefixes=LIB %s

# RUN: ld.lld -Tlinker.t -z undefs symbols.32.o abs.32.o -o symbols.32
# RUN: ld.lld -Tlinker.t -z undefs symbols.64.o abs.64.o -o symbols.64
# RUN: llvm-objdump -d --no-show-raw-insn symbols.32 | FileCheck --check-prefixes=EXE %s
# RUN: llvm-objdump -d --no-show-raw-insn symbols.64 | FileCheck --check-prefixes=EXE %s


## Symbol 'hidden_sym' is nonpreemptible, the relaxation should be applied.
LIB:      pcaddi      $a0, [[#]]
## Symbol 'global_sym' is preemptible, no relaxations should be applied.
LIB-NEXT: pcalau12i   $a1, 4
LIB-NEXT: ld.{{[wd]}} $a1, $a1, [[#]]
## Symbol 'undefined_sym' is undefined, no relaxations should be applied.
LIB-NEXT: pcalau12i   $a2, 4
LIB-NEXT: ld.{{[wd]}} $a2, $a2, [[#]]
## Symbol 'ifunc_sym' is STT_GNU_IFUNC, no relaxations should be applied.
LIB-NEXT: pcalau12i   $a3, 4
LIB-NEXT: ld.{{[wd]}} $a3, $a3, [[#]]
## Symbol 'abs_sym' is absolute, no relaxations should be applied.
LIB-NEXT: pcalau12i   $a4, 4
LIB-NEXT: ld.{{[wd]}} $a4, $a4, [[#]]


## Symbol 'hidden_sym' is nonpreemptible, the relaxation should be applied.
EXE:      pcaddi      $a0, [[#]]
## Symbol 'global_sym' is nonpreemptible, the relaxation should be applied.
EXE-NEXT: pcaddi      $a1, [[#]]
## Symbol 'undefined_sym' is undefined, no relaxations should be applied.
EXE-NEXT: pcalau12i   $a2, 4
EXE-NEXT: ld.{{[wd]}} $a2, $a2, [[#]]
## Symbol 'ifunc_sym' is STT_GNU_IFUNC, no relaxations should be applied.
EXE-NEXT: pcalau12i   $a3, 4
EXE-NEXT: ld.{{[wd]}} $a3, $a3, [[#]]
## Symbol 'abs_sym' is absolute, relaxations may be applied in -no-pie mode.
EXE-NEXT: pcaddi      $a4, -[[#]]


## The linker script ensures that .rodata and .text are near (>4M) so that
## the pcalau12i+ld.w/d pair can be relaxed to pcaddi.
#--- linker.t
SECTIONS {
 .text   0x10000: { *(.text) }
 .rodata 0x14000: { *(.rodata) }
}

# This symbol is defined in a separate file to prevent the definition from
# being folded into the instructions that reference it.
#--- abs.s
.global abs_sym
.hidden abs_sym
abs_sym = 0x1000

#--- symbols.s
.rodata
.hidden hidden_sym
hidden_sym:
.word 10

.global global_sym
global_sym:
.word 10

.text
.type ifunc_sym STT_GNU_IFUNC
.hidden ifunc_sym
ifunc_sym:
  nop

.global _start
_start:
  la.got    $a0, hidden_sym
  la.got    $a1, global_sym
  la.got    $a2, undefined_sym
  la.got    $a3, ifunc_sym
  la.got    $a4, abs_sym
