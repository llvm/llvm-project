# REQUIRES: mips
# Check handling TLS related relocations and symbols when linking
# a 64-bit static executable.

# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux %s -o %t
# RUN: ld.lld -static %t -o %t.exe
# RUN: llvm-objdump -s -t %t.exe | FileCheck %s

# CHECK: SYMBOL TABLE:
# CHECK: [[TGA:[0-9a-f]{8}]] g      .text  0000000000000000 __tls_get_addr
# CHECK:    0000000000000000 g      .tdata 0000000000000000 tls1
#
# CHECK:      Contents of section .data:
# CHECK-NEXT:  {{.*}} [[TGA]] ffffffff ffff8004 ffffffff
# CHECK-NEXT:  {{.*}} ffff9004

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
  .dtpreldword tls1+4   # R_MIPS_TLS_DTPREL64
  .tpreldword tls1+4    # R_MIPS_TLS_TPREL64

 .section .tdata,"awT",%progbits
 .global tls1
tls1:
 .word __tls_get_addr
 .word 0
