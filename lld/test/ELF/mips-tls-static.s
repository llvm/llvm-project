# Check handling TLS related relocations and symbols when linking
# a static executable.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t
# RUN: ld.lld -static %t -o %t.exe
# RUN: llvm-objdump -s -t %t.exe | FileCheck %s

# REQUIRES: mips

# CHECK:      Contents of section .data:
# CHECK-NEXT:  40000 00020004 ffff8004 ffff9004
#
# CHECK: SYMBOL TABLE:
# CHECK: 00020004         .text           00000000 __tls_get_addr
# CHECK: 00000000 g       .tdata          00000000 tls1

  .text
  .global __start
__start:
  nop

  .global __tls_get_addr
__tls_get_addr:
  nop

  .data
loc:
  .word __tls_get_addr
  .dtprelword tls1+4    # R_MIPS_TLS_DTPREL32
  .tprelword tls1+4     # R_MIPS_TLS_TPREL32

 .section .tdata,"awT",%progbits
 .global tls1
tls1:
 .word __tls_get_addr
 .word 0
