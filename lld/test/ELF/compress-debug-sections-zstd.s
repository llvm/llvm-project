# REQUIRES: x86, zstd

# RUN: llvm-mc -filetype=obj -triple=x86_64 --compress-debug-sections=zstd %s -o %t.o

# RUN: ld.lld %t.o -o %t.so -shared
# RUN: llvm-readelf -S -x .debug_str %t.so | FileCheck %s

# CHECK:      .debug_str    PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 01 MS  0 0  1
# CHECK:      Hex dump of section '.debug_str':
# CHECK-NEXT: 0x00000000 73686f72 7420756e 7369676e 65642069 short unsigned i
# CHECK-NEXT: 0x00000010 6e740075 6e736967 6e656420 63686172 nt.unsigned char
# CHECK-NEXT: 0x00000020 00636861 72006c6f 6e672075 6e736967 .char.long unsig
# CHECK-NEXT: 0x00000030 6e656420 696e7400 756e7369 676e6564 ned int.unsigned
# CHECK-NEXT: 0x00000040 20696e74 00                          int.

# RUN: ld.lld %t.o -o %t.so -shared --compress-debug-sections=zstd
# RUN: llvm-readelf -S %t.so | FileCheck %s --check-prefix=OUTPUT-SEC
# RUN: llvm-objcopy --decompress-debug-sections %t.so
# RUN: llvm-readelf -S -x .debug_str %t.so | FileCheck %s

# OUTPUT-SEC: .debug_str    PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 01 MSC 0 0  1

.section .debug_str,"MS",@progbits,1
.LASF2:
 .string "short unsigned int"
.LASF3:
 .string "unsigned int"
.LASF0:
 .string "long unsigned int"
.LASF8:
 .string "char"
.LASF1:
 .string "unsigned char"
