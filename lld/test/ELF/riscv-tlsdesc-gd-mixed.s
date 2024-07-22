# REQUIRES: riscv
# RUN: llvm-mc -filetype=obj -triple=riscv64 %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readobj -r %t.so | FileCheck %s --check-prefix=RELA

## Both TLSDESC and DTPMOD64/DTPREL64 should be present.
# RELA:      .rela.dyn {
# RELA-NEXT:   0x[[#%X,ADDR:]] R_RISCV_TLSDESC      a 0x0
# RELA-NEXT:   0x[[#ADDR+16]]  R_RISCV_TLS_DTPMOD64 a 0x0
# RELA-NEXT:   0x[[#ADDR+24]]  R_RISCV_TLS_DTPREL64 a 0x0
# RELA-NEXT: }

  la.tls.gd a0,a
  call __tls_get_addr@plt

.Ltlsdesc_hi0:
  auipc a2, %tlsdesc_hi(a)
  ld    a3, %tlsdesc_load_lo(.Ltlsdesc_hi0)(a2)
  addi  a0, a2, %tlsdesc_add_lo(.Ltlsdesc_hi0)
  jalr  t0, 0(a3), %tlsdesc_call(.Ltlsdesc_hi0)

.section .tbss,"awT",@nobits
.globl a
.zero 8
a:
.zero 4
