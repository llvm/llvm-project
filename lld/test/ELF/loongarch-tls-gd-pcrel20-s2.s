# REQUIRES: loongarch
# RUN: rm -rf %t && split-file %s %t

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
# GD32-REL-NEXT:   0x20300 R_LARCH_TLS_DTPMOD32 a 0x0
# GD32-REL-NEXT:   0x20304 R_LARCH_TLS_DTPREL32 a 0x0
# GD32-REL-NEXT:   0x20308 R_LARCH_TLS_DTPMOD32 b 0x0
# GD32-REL-NEXT:   0x2030C R_LARCH_TLS_DTPREL32 b 0x0
# GD32-REL-NEXT: }

## &DTPMOD(a) - . = 0x20300 - 0x10250 = 16428<<2
# GD32:      10250: pcaddi $a0, 16428
# GD32-NEXT:        bl 44

## &DTPMOD(b) - . = 0x20308 - 0x10258 = 16428<<2
# GD32:      10258: pcaddi $a0, 16428
# GD32-NEXT:        bl 36

# GD64-REL:      .rela.dyn {
# GD64-REL-NEXT:   0x204C0 R_LARCH_TLS_DTPMOD64 a 0x0
# GD64-REL-NEXT:   0x204C8 R_LARCH_TLS_DTPREL64 a 0x0
# GD64-REL-NEXT:   0x204D0 R_LARCH_TLS_DTPMOD64 b 0x0
# GD64-REL-NEXT:   0x204D8 R_LARCH_TLS_DTPREL64 b 0x0
# GD64-REL-NEXT: }

## &DTPMOD(a) - . = 0x204c0 - 0x10398 = 16458<<2
# GD64:      10398: pcaddi $a0, 16458
# GD64-NEXT:        bl 52

## &DTPMOD(b) - . = 0x204d0 - 0x103a4 = 16460<<2
# GD64:      103a0: pcaddi $a0, 16460
# GD64-NEXT:        bl 44

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
# IE32-REL-NEXT:   0x30220 R_LARCH_TLS_DTPMOD32 b 0x0
# IE32-REL-NEXT:   0x30224 R_LARCH_TLS_DTPREL32 b 0x0
# IE32-REL-NEXT: }
# IE32-GOT:      section '.got':
# IE32-GOT-NEXT: 0x00030218 01000000 08000000 00000000 00000000

# IE64-REL:      .rela.dyn {
# IE64-REL-NEXT:   0x30380 R_LARCH_TLS_DTPMOD64 b 0x0
# IE64-REL-NEXT:   0x30388 R_LARCH_TLS_DTPREL64 b 0x0
# IE64-REL-NEXT: }
# IE64-GOT:      section '.got':
# IE64-GOT-NEXT: 0x00030370 01000000 00000000 08000000 00000000
# IE64-GOT-NEXT: 0x00030380 00000000 00000000 00000000 00000000

#--- a.s
pcaddi $a0, %gd_pcrel_20(a)
bl %plt(__tls_get_addr)

pcaddi $a0, %gd_pcrel_20(b)
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
