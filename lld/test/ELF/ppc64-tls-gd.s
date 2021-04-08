# REQUIRES: ppc
# RUN: llvm-mc -filetype=obj -triple=powerpc64le %s -o %t.o
# RUN: echo '.tbss; .globl b, c; b: .zero 4; c:' | llvm-mc -filetype=obj -triple=powerpc64le - -o %t1.o
# RUN: ld.lld -shared -soname=t1.so %t1.o -o %t1.so

# RUN: ld.lld -shared %t.o %t1.o -o %t.so
# RUN: llvm-readobj -r %t.so | FileCheck --check-prefix=GD-REL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t.so | FileCheck --check-prefix=GD %s

# RUN: ld.lld %t.o %t1.o -o %t
# RUN: llvm-readelf -r %t | FileCheck --check-prefix=NOREL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=LE %s

# RUN: ld.lld %t.o %t1.so -o %t
# RUN: llvm-readobj -r %t | FileCheck --check-prefix=IE-REL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=IE %s

# GD-REL:      .rela.dyn {
# GD-REL-NEXT:   0x20550 R_PPC64_DTPMOD64 a 0x0
# GD-REL-NEXT:   0x20558 R_PPC64_DTPREL64 a 0x0
# GD-REL-NEXT:   0x20560 R_PPC64_DTPMOD64 b 0x0
# GD-REL-NEXT:   0x20568 R_PPC64_DTPREL64 b 0x0
# GD-REL-NEXT:   0x20570 R_PPC64_DTPMOD64 c 0x0
# GD-REL-NEXT:   0x20578 R_PPC64_DTPREL64 c 0x0
# GD-REL-NEXT: }

## Start with .got[1] as .got[0] is the .TOC.
## &DTPMOD(a) - .TOC. = &.got[1] - (.got+0x8000) = -32760
# GD:      addis 3, 2, 0
# GD-NEXT: addi 3, 3, -32760
# GD-NEXT: bl 0x10400
# GD-NEXT: ld 2, 24(1)

## &DTPMOD(b) - .TOC. = &.got[3] - (.got+0x8000) = -32744
# GD-NEXT: addis 3, 2, 0
# GD-NEXT: addi 3, 3, -32744
# GD-NEXT: bl 0x10400
# GD-NEXT: ld 2, 24(1)

## &DTPMOD(b) - .TOC. = &.got[5] - (.got+0x8000) = -32728
# GD-NEXT: li 3, -32728
# GD-NEXT: bl 0x10400
# GD-NEXT: ld 2, 24(1)

# NOREL: no relocations

## a@tprel = st_value(a)-0x7000 = -28664
# LE:      nop
# LE-NEXT: addis 3, 13, 0
# LE-NEXT: nop
# LE-NEXT: addi 3, 3, -28664
## b@tprel = st_value(b)-0x7000 = -28660
# LE:      nop
# LE-NEXT: addis 3, 13, 0
# LE-NEXT: nop
# LE-NEXT: addi 3, 3, -28660
## c@tprel = st_value(c)-0x7000 = -28656
# LE-NEXT: addis 3, 13, 0
# LE-NEXT: nop
# LE-NEXT: addi 3, 3, -28656

# IE-REL:      .rela.dyn {
# IE-REL-NEXT:   0x10020420 R_PPC64_TPREL64 b 0x0
# IE-REL-NEXT:   0x10020428 R_PPC64_TPREL64 c 0x0
# IE-REL-NEXT: }

## a is relaxed to use LE.
## a@tprel = st_value(a)-0x7000 = -28664
# IE:      nop
# IE-NEXT: addis 3, 13, 0
# IE-NEXT: nop
# IE-NEXT: addi 3, 3, -28664
## &DTPMOD(b) - .TOC. = &.got[1] - (.got+0x8000) = -32760
# IE-NEXT: addis 3, 2, 0
# IE-NEXT: ld 3, -32760(3)
# IE-NEXT: nop
# IE-NEXT: add 3, 3, 13
## &DTPMOD(c) - .TOC. = &.got[2] - (.got+0x8000) = -32752
## r0 is wrong. R_PPC64_GOT_TLS16 cannot be relaxed to IE but the behavior is
## consistent with ld.bfd
# IE-NEXT: ld 3, -32752(0)
# IE-NEXT: nop
# IE-NEXT: add 3, 3, 13

addis 3, 2, a@got@tlsgd@ha
addi 3, 3, a@got@tlsgd@l
bl __tls_get_addr(a@tlsgd)
nop

addis 3, 2, b@got@tlsgd@ha
addi 3, 3, b@got@tlsgd@l
bl __tls_get_addr(b@tlsgd)
nop

addi 3, 0, c@got@tlsgd
bl __tls_get_addr(c@tlsgd)
nop

.section .tbss
.globl a
.zero 8
a:
.zero 4
