## This test verifies that the pair adrp + ldr is relaxed/not relaxed
## depending on the target symbol properties.

# REQUIRES: aarch64
# RUN: rm -rf %t && split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=aarch64 %t/symbols.s -o %t/symbols.o

# RUN: ld.lld -shared -T %t/linker.t %t/symbols.o -o %t/symbols.so
# RUN: llvm-objdump --no-show-raw-insn -d %t/symbols.so | \
# RUN:   FileCheck --check-prefix=LIB %s

## Symbol 'hidden_sym' is nonpreemptible, the relaxation should be applied.
LIB:      adrp   x0
LIB-NEXT: add    x0

## Symbol 'global_sym' is preemptible, no relaxations should be applied.
LIB-NEXT: adrp   x1
LIB-NEXT: ldr    x1

## Symbol 'undefined_sym' is undefined, no relaxations should be applied.
LIB-NEXT: adrp   x2
LIB-NEXT: ldr    x2

## Symbol 'ifunc_sym' is STT_GNU_IFUNC, no relaxations should be applied.
LIB-NEXT: adrp   x3
LIB-NEXT: ldr    x3

# RUN: ld.lld -T %t/linker.t -z undefs %t/symbols.o -o %t/symbols
# RUN: llvm-objdump --no-show-raw-insn -d %t/symbols | \
# RUN:   FileCheck --check-prefix=EXE %s

## Symbol 'global_sym' is nonpreemptible, the relaxation should be applied.
EXE:      adrp   x1
EXE-NEXT: add    x1

## The linker script ensures that .rodata and .text are sufficiently (>1MB)
## far apart so that the adrp + ldr pair cannot be relaxed to adr + nop.
#--- linker.t
SECTIONS {
 .rodata 0x1000: { *(.rodata) }
 .text   0x300100: { *(.text) }
}

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
  adrp    x0, :got:hidden_sym
  ldr     x0, [x0, #:got_lo12:hidden_sym]
  adrp    x1, :got:global_sym
  ldr     x1, [x1, #:got_lo12:global_sym]
  adrp    x2, :got:undefined_sym
  ldr     x2, [x2, #:got_lo12:undefined_sym]
  adrp    x3, :got:ifunc_sym
  ldr     x3, [x3, #:got_lo12:ifunc_sym]
