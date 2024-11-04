# REQUIRES: x86, zstd

# RUN: llvm-mc -filetype=obj -triple=x86_64 --compress-debug-sections=zstd %s -o %t.o

# RUN: ld.lld %t.o -o %t.so -shared
# RUN: llvm-readelf -S -p .debug_str %t.so | FileCheck %s

# CHECK:      .debug_str    PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 01 MS  0 0  1
# CHECK:      String dump of section '.debug_str':
# CHECK-NEXT: [     0] {{A+}}
# CHECK-NEXT: [    81] short unsigned int
# CHECK-NEXT: [    94] unsigned char
# CHECK-NEXT: [    a2] char
# CHECK-NEXT: [    a7] long unsigned int
# CHECK-NEXT: [    b9] unsigned int

# RUN: ld.lld %t.o -o %t.so -shared --compress-debug-sections=zstd
# RUN: llvm-readelf -S %t.so | FileCheck %s --check-prefix=OUTPUT-SEC
# RUN: llvm-objcopy --decompress-debug-sections %t.so
# RUN: llvm-readelf -S -p .debug_str %t.so | FileCheck %s

# OUTPUT-SEC:      .debug_str    PROGBITS [[#%x,]] [[#%x,]] [[#%x,]] 01 MSC 0 0  1
# OUTPUT-SEC-NEXT: .debug_frame  PROGBITS [[#%x,]] [[#%x,]] 000000   00     0 0  1
# OUTPUT-SEC-NEXT: .debug_loc    PROGBITS [[#%x,]] [[#%x,]] 000010   00     0 0  1

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
.Lunused:
 .fill 128, 1, 0x41
 .byte 0

## Test sections where compressed content would be larger.
.section .debug_frame,""
.section .debug_loc,""
.space 16
