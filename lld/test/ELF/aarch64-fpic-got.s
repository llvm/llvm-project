# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux %p/Inputs/shared.s -o %t-lib.o
# RUN: ld.lld -shared %t-lib.o -soname t-lib.so -o %t-lib.so

# RUN: ld.lld %t-lib.so %t.o -o %t.exe
# RUN: llvm-readobj -r %t.exe | FileCheck --check-prefix=RELOC %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.exe | FileCheck --check-prefix=DIS %s

## Checks if got access to dynamic objects is done through a got relative
## dynamic relocation and not using plt relative (R_AARCH64_JUMP_SLOT).
# RELOC:      .rela.dyn {
# RELOC-NEXT:   0x220320 R_AARCH64_GLOB_DAT bar 0x0
# RELOC-NEXT: }

## page(0x220320) - page(0x210000) = 65536
## page(0x220320) & 0xff8 = 800
# DIS:      <_start>:
# DIS-NEXT: 210258: adrp x0, 0x220000
# DIS-NEXT: 21025c: ldr x0, [x0, #800]

.globl _start
_start:
  adrp x0, :got:bar
  ldr  x0, [x0, :got_lo12:bar]
