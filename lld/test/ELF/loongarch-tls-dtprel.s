# REQUIRES: loongarch
# RUN: llvm-mc -filetype=obj -triple=loongarch32 %s -o %32.o
# RUN: llvm-mc -filetype=obj -triple=loongarch64 %s -o %64.o
# RUN: llvm-readobj -r %32.o | FileCheck %s
# RUN: llvm-readobj -r %64.o | FileCheck %s
# RUN: ld.lld %32.o -o %32
# RUN: ld.lld %64.o -o %64

# CHECK:      .rela.debug_info {
# CHECK-NEXT:   0x0 R_LARCH_TLS_DTPREL32 var 0x0
# CHECK-NEXT:   0x4 R_LARCH_TLS_DTPREL32 .tdata 0x1
# CHECK-NEXT:   0x8 R_LARCH_TLS_DTPREL64 var 0x0
# CHECK-NEXT:   0x10 R_LARCH_TLS_DTPREL64 .tdata 0x1
# CHECK-NEXT: }

.section .tdata,"awT",@progbits
.skip 8
.globl var
var:
  .word 0

.section        .debug_info,"",@progbits
  .dtprelword var
  .dtprelword .tdata+1
  .dtpreldword var
  .dtpreldword .tdata+1
