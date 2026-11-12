# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
# RUN: llvm-readobj -r %t.o | FileCheck %s
# RUN: ld.lld %t.o -o %t

# CHECK:      .rela.debug_info {
# CHECK-NEXT:   0x0 R_AARCH64_TLS_DTPREL64 var 0x0
# CHECK-NEXT:   0x8 R_AARCH64_TLS_DTPREL64 var 0x1
# CHECK-NEXT:   0x10 R_AARCH64_TLS_DTPREL64 .tdata 0x0
# CHECK-NEXT:   0x18 R_AARCH64_TLS_DTPREL64 .tdata 0x1
# CHECK-NEXT: }

.section .tdata,"awT",@progbits
.skip 8
.globl var
var:
  .word 0

.section        .debug_info,"",@progbits
  .xword  %dtprel(var)
  .xword  %dtprel(var+1)
  .xword  %dtprel(.tdata)
  .xword  %dtprel(.tdata+1)
