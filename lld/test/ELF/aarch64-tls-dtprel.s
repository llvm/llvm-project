# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck %s
# RUN: ld.lld %t.o -o %t

# CHECK:      .rela.debug_info {
# CHECK-NEXT:   0x0 R_AARCH64_TLS_DTPREL64 var 0x0
# CHECK-NEXT:   0x8 R_AARCH64_TLS_DTPREL64 var 0x1
# CHECK-NEXT: }

.section .tdata,"awT",@progbits
.globl var
var:
  .word 0

.section        .debug_info,"",@progbits
  .xword  %dtprel(var)
  .xword  %dtprel(var+1)
