# REQUIRES: x86, zstd

# RUN: llvm-mc -filetype=obj -triple=x86_64 --compress-debug-sections=zstd %s -o %t.o

# RUN: ld.lld %t.o -o %t.so -shared
# RUN: llvm-readelf -S -x .debug_str %t.so | FileCheck %s

# CHECK:      .debug_str    PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 01 MS  0 0  1
# CHECK:      Hex dump of section '.debug_str':
# CHECK-NEXT: 0x00000000 756e7369 676e6564 20696e74 00636861 unsigned int.cha
# CHECK-NEXT: 0x00000010 7200756e 7369676e 65642063 68617200 r.unsigned char.
# CHECK-NEXT: 0x00000020 73686f72 7420756e 7369676e 65642069 short unsigned i
# CHECK-NEXT: 0x00000030 6e74006c 6f6e6720 756e7369 676e6564 nt.long unsigned
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
