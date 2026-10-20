# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=aarch64 %p/Inputs/shared.s -o lib.o
# RUN: ld.lld -shared lib.o -soname lib.so -o lib.so

## Checks if got access to dynamic objects is done through a got relative
## dynamic relocation and not using plt relative (R_AARCH64_JUMP_SLOT).
# RELOC:      .rela.dyn {
# RELOC-NEXT:   0x220318 R_AARCH64_GLOB_DAT bar 0x0
# RELOC-NEXT: }

#--- small.s

# RUN: llvm-mc -filetype=obj -triple=aarch64 small.s -o small.o
# RUN: ld.lld lib.so small.o -o small
# RUN: llvm-readobj -r small | FileCheck --check-prefix=RELOC %s
# RUN: llvm-objdump -d --no-show-raw-insn small | FileCheck --check-prefix=DIS-SMALL %s

## page(0x220318) & 0xff8 = 0x318
# DIS-SMALL:      <_start>:
# DIS-SMALL-NEXT: adrp x0, 0x220000
# DIS-SMALL-NEXT: ldr x0, [x0, #0x318]

.globl _start
_start:
  adrp x0, :got:bar
  ldr  x0, [x0, :got_lo12:bar]

#--- tiny.s

# RUN: llvm-mc -filetype=obj -triple=aarch64 tiny.s -o tiny.o
# RUN: ld.lld lib.so tiny.o -o tiny
# RUN: llvm-readobj -r tiny | FileCheck --check-prefix=RELOC %s
# RUN: llvm-objdump -d --no-show-raw-insn tiny | FileCheck --check-prefix=DIS-TINY %s

# DIS-TINY:      <_start>:
# DIS-TINY-NEXT: ldr x0, 0x220318

.globl _start
_start:
  ldr  x0, :got:bar
