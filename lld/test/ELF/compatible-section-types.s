# REQUIRES: x86

## Some section types (e.g. SHT_NOBITS, SHT_NOTE, SHT_PREINIT_ARRAY) can be
## mixed with SHT_PROGBITS. The output type is SHT_PROGBITS.

# RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
# RUN: ld.lld -shared %t.o -o %t
# RUN: llvm-readelf -S %t | FileCheck %s

# CHECK: .foo2 PROGBITS {{.*}} 000002 00 A
# CHECK: .foo  PROGBITS {{.*}} 000028 00 WA
# CHECK: .foo1 PROGBITS {{.*}} 000010 00 WA

.section .foo, "aw", @progbits, unique, 1
.quad 0

.section .foo, "aw", @init_array, unique, 2
.quad 0

.section .foo, "aw", @preinit_array, unique, 3
.quad 0

.section .foo, "aw", @fini_array, unique, 4
.quad 0

.section .foo, "aw", @note, unique, 5
.quad 0

.section .foo1, "aw", @progbits, unique, 1
.quad 0
.section .foo1, "aw", @nobits, unique, 2
.quad 0

.section .foo2, "a", @nobits, unique, 1
.byte 0
.section .foo2, "a", @progbits, unique, 2
.byte 0
