# REQUIRES: loongarch
# RUN: llvm-mc -filetype=obj -triple=loongarch64 %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readobj -r %t.so | FileCheck %s --check-prefix=RELA

## Both TLSDESC and DTPMOD64/DTPREL64 should be present.
# RELA:      .rela.dyn {
# RELA-NEXT:   0x[[#%X,ADDR:]] R_LARCH_TLS_DESC64   a 0x0
# RELA-NEXT:   0x[[#ADDR+16]]  R_LARCH_TLS_DTPMOD64 a 0x0
# RELA-NEXT:   0x[[#ADDR+24]]  R_LARCH_TLS_DTPREL64 a 0x0
# RELA-NEXT: }

  la.tls.gd $a0,a
  bl %plt(__tls_get_addr)

  la.tls.desc $a0, a
  add.d $a1, $a0, $tp

.section .tbss,"awT",@nobits
.globl a
.zero 8
a:
.zero 4
