# REQUIRES: loongarch
# RUN: rm -rf %t && split-file %s %t

## LoongArch psABI doesn't specify TLS relaxation. Though the code sequences are not
## relaxed, dynamic relocations can be omitted for GD->LE relaxation.

# RUN: llvm-mc --filetype=obj --triple=loongarch32 %t/a.s -o %t/a.32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch32 %t/bc.s -o %t/bc.32.o
# RUN: ld.lld -shared -soname=bc.so %t/bc.32.o -o %t/bc.32.so
# RUN: llvm-mc --filetype=obj --triple=loongarch32 %t/tga.s -o %t/tga.32.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 %t/a.s -o %t/a.64.o
# RUN: llvm-mc --filetype=obj --triple=loongarch64 %t/bc.s -o %t/bc.64.o
# RUN: ld.lld -shared -soname=bc.so %t/bc.64.o -o %t/bc.64.so
# RUN: llvm-mc --filetype=obj --triple=loongarch64 %t/tga.s -o %t/tga.64.o

## LA32 GD
# RUN: ld.lld -shared %t/a.32.o %t/bc.32.o -o %t/gd.32.so
# RUN: llvm-readobj -r %t/gd.32.so | FileCheck --check-prefix=GD32-REL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t/gd.32.so | FileCheck --check-prefix=GD32 %s

## LA32 GD -> LE
# RUN: ld.lld %t/a.32.o %t/bc.32.o %t/tga.32.o -o %t/le.32
# RUN: llvm-readelf -r %t/le.32 | FileCheck --check-prefix=NOREL %s
# RUN: llvm-readelf -x .got %t/le.32 | FileCheck --check-prefix=LE32-GOT %s
# RUN: ld.lld -pie %t/a.32.o %t/bc.32.o %t/tga.32.o -o %t/le-pie.32
# RUN: llvm-readelf -r %t/le-pie.32 | FileCheck --check-prefix=NOREL %s
# RUN: llvm-readelf -x .got %t/le-pie.32 | FileCheck --check-prefix=LE32-GOT %s

## LA32 GD -> IE
# RUN: ld.lld %t/a.32.o %t/bc.32.so %t/tga.32.o -o %t/ie.32
# RUN: llvm-readobj -r %t/ie.32 | FileCheck --check-prefix=IE32-REL %s
# RUN: llvm-readelf -x .got %t/ie.32 | FileCheck --check-prefix=IE32-GOT %s

## LA64 GD
# RUN: ld.lld -shared %t/a.64.o %t/bc.64.o -o %t/gd.64.so
# RUN: llvm-readobj -r %t/gd.64.so | FileCheck --check-prefix=GD64-REL %s
# RUN: llvm-objdump -d --no-show-raw-insn %t/gd.64.so | FileCheck --check-prefix=GD64 %s

## LA64 GD -> LE
# RUN: ld.lld %t/a.64.o %t/bc.64.o %t/tga.64.o -o %t/le.64
# RUN: llvm-readelf -r %t/le.64 | FileCheck --check-prefix=NOREL %s
# RUN: llvm-readelf -x .got %t/le.64 | FileCheck --check-prefix=LE64-GOT %s
# RUN: ld.lld -pie %t/a.64.o %t/bc.64.o %t/tga.64.o -o %t/le-pie.64
# RUN: llvm-readelf -r %t/le-pie.64 | FileCheck --check-prefix=NOREL %s
# RUN: llvm-readelf -x .got %t/le-pie.64 | FileCheck --check-prefix=LE64-GOT %s

## LA64 GD -> IE
# RUN: ld.lld %t/a.64.o %t/bc.64.so %t/tga.64.o -o %t/ie.64
# RUN: llvm-readobj -r %t/ie.64 | FileCheck --check-prefix=IE64-REL %s
# RUN: llvm-readelf -x .got %t/ie.64 | FileCheck --check-prefix=IE64-GOT %s

# GD32-REL:      .rela.dyn {
# GD32-REL-NEXT:   0x20310 R_LARCH_TLS_DTPMOD32 a 0x0
# GD32-REL-NEXT:   0x20314 R_LARCH_TLS_DTPREL32 a 0x0
# GD32-REL-NEXT:   0x20318 R_LARCH_TLS_DTPMOD32 b 0x0
# GD32-REL-NEXT:   0x2031C R_LARCH_TLS_DTPREL32 b 0x0
# GD32-REL-NEXT: }

## &DTPMOD(a) - . = 0x20310 - 0x10250: 0x10 pages, page offset 0x310
# GD32:      10250: pcalau12i $a0, 16
# GD32-NEXT:        addi.w $a0, $a0, 784
# GD32-NEXT:        bl 56

## &DTPMOD(b) - . = 0x20318 - 0x1025c: 0x10 pages, page offset 0x318
# GD32:      1025c: pcalau12i $a0, 16
# GD32-NEXT:        addi.w $a0, $a0, 792
# GD32-NEXT:        bl 44

# GD64-REL:      .rela.dyn {
# GD64-REL-NEXT:   0x204C0 R_LARCH_TLS_DTPMOD64 a 0x0
# GD64-REL-NEXT:   0x204C8 R_LARCH_TLS_DTPREL64 a 0x0
# GD64-REL-NEXT:   0x204D0 R_LARCH_TLS_DTPMOD64 b 0x0
# GD64-REL-NEXT:   0x204D8 R_LARCH_TLS_DTPREL64 b 0x0
# GD64-REL-NEXT: }

## &DTPMOD(a) - . = 0x204c0 - 0x10398: 0x10 pages, page offset 0x4c0
# GD64:      10398: pcalau12i $a0, 16
# GD64-NEXT:        addi.d $a0, $a0, 1216
# GD64-NEXT:        bl 48

## &DTPMOD(b) - . = 0x204d0 - 0x103a4: 0x10 pages, page offset 0x4d0
# GD64:      103a4: pcalau12i $a0, 16
# GD64-NEXT:        addi.d $a0, $a0, 1232
# GD64-NEXT:        bl 36

# NOREL: no relocations

## .got contains pre-populated values: [a@dtpmod, a@dtprel, b@dtpmod, b@dtprel]
## a@dtprel = st_value(a) = 0x8
## b@dtprel = st_value(b) = 0xc
# LE32-GOT: section '.got':
# LE32-GOT-NEXT: 0x[[#%x,A:]] 01000000 08000000 01000000 0c000000
# LE64-GOT: section '.got':
# LE64-GOT-NEXT: 0x[[#%x,A:]] 01000000 00000000 08000000 00000000
# LE64-GOT-NEXT: 0x[[#%x,A:]] 01000000 00000000 0c000000 00000000

## a is local - relaxed to LE - its DTPMOD/DTPREL slots are link-time constants.
## b is external - DTPMOD/DTPREL dynamic relocations are required.
# IE32-REL:      .rela.dyn {
# IE32-REL-NEXT:   0x30228 R_LARCH_TLS_DTPMOD32 b 0x0
# IE32-REL-NEXT:   0x3022C R_LARCH_TLS_DTPREL32 b 0x0
# IE32-REL-NEXT: }
# IE32-GOT:      section '.got':
# IE32-GOT-NEXT: 0x00030220 01000000 08000000 00000000 00000000

# IE64-REL:      .rela.dyn {
# IE64-REL-NEXT:   0x30388 R_LARCH_TLS_DTPMOD64 b 0x0
# IE64-REL-NEXT:   0x30390 R_LARCH_TLS_DTPREL64 b 0x0
# IE64-REL-NEXT: }
# IE64-GOT:      section '.got':
# IE64-GOT-NEXT: 0x00030378 01000000 00000000 08000000 00000000
# IE64-GOT-NEXT: 0x00030388 00000000 00000000 00000000 00000000

#--- a.s
la.tls.gd $a0, a
bl %plt(__tls_get_addr)

la.tls.gd $a0, b
bl %plt(__tls_get_addr)

.section .tbss,"awT",@nobits
.globl a
.zero 8
a:
.zero 4

#--- bc.s
.section .tbss,"awT",@nobits
.globl b, c
b:
.zero 4
c:

#--- tga.s
.globl __tls_get_addr
__tls_get_addr:
