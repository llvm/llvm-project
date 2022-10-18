# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readobj -d -r %t.so | FileCheck %s --check-prefix=IE-REL
# RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t.so | FileCheck %s --check-prefix=IE

# RUN: ld.lld %t.o -o %t
# RUN: llvm-readobj -d -r %t | FileCheck %s --check-prefix=LE-REL
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s --check-prefix=LE

# IE-REL:      FLAGS STATIC_TLS
# IE-REL:      .rela.dyn {
# IE-REL-NEXT:   0x20390 R_AARCH64_TLS_TPREL64 - 0xC
# IE-REL-NEXT:   0x20388 R_AARCH64_TLS_TPREL64 a 0x0
# IE-REL-NEXT: }

# IE:          adrp    x0, 0x20000
# IE-NEXT:     ldr     x0, [x0, #0x388]
# IE-NEXT:     adrp    x1, 0x20000
# IE-NEXT:     ldr     x1, [x1, #0x390]

# LE-REL-NOT:  FLAGS
# LE-REL:      Relocations [
# LE-REL-NEXT: ]

## TP is followed by a gap of 2 words, followed by alignment padding (empty in this case), then the static TLS blocks.
## a's offset is 16+8=24.
# LE:          movz    x0, #0, lsl #16
# LE-NEXT:     movk    x0, #24
# LE-NEXT:     movz    x1, #0, lsl #16
# LE-NEXT:     movk    x1, #28

.globl _start
_start:
  adrp x0, :gottprel:a
  ldr x0, [x0, #:gottprel_lo12:a]
  adrp x1, :gottprel:b
  ldr x1, [x1, #:gottprel_lo12:b]

.section .tbss,"awT",%nobits
.globl a
.zero 8
a:
.zero 4
b:
