# REQUIRES: x86, zlib
## Test SHF_ALLOC|SHF_COMPRESSED input sections, which llvm-mc may produce.

# RUN: llvm-mc -filetype=obj -triple=x86_64 --compress-debug-sections=zlib %s -o %t.64.o
# RUN: llvm-mc -filetype=obj -triple=i386 --compress-debug-sections=zlib %s -o %t.32.o
# RUN: llvm-readelf -S %t.64.o | FileCheck %s --check-prefix=INPUT
# INPUT: .debug_gdb_scripts PROGBITS {{.*}} AC

## For i386, the implicit addend is read from the decompressed content.
# RUN: ld.lld --defsym=_start=0x10000 %t.64.o -o %t.64
# RUN: llvm-readelf -x .debug_gdb_scripts %t.64 | FileCheck %s
# RUN: ld.lld --defsym=_start=0x10000 %t.32.o -o %t.32
# RUN: llvm-readelf -x .debug_gdb_scripts %t.32 | FileCheck %s

# CHECK:      Hex dump of section '.debug_gdb_scripts':
# CHECK-NEXT: 0x{{.*}} 34120100 61623000 61623100 61623200 4...ab0.ab1.ab2.
# CHECK-NEXT: 0x{{.*}} 61623300 61623400 61623500 61623600 ab3.ab4.ab5.ab6.
# CHECK-NEXT: 0x{{.*}} 61623700 61623800 61623900 61623130 ab7.ab8.ab9.ab10
# CHECK-NEXT: 0x{{.*}} 00616231 31006162 31320061 62313300 .ab11.ab12.ab13.
# CHECK-NEXT: 0x{{.*}} 61623134 00616231 3500              ab14.ab15.
# CHECK-EMPTY:

.section .debug_gdb_scripts,"a",@progbits
.long _start+0x1234
.rept 16
.asciz "ab\+"
.endr
