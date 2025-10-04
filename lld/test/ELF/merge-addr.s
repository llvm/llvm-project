# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -o %t.so -shared --section-start cst4=0x800 --section-start str=0x1000
# RUN: llvm-readelf -r -S %t.so | FileCheck %s
# RUN: llvm-objdump -s %t.so | FileCheck %s --check-prefix=OBJDUMP

# RUN: ld.lld %t.o -o %t0.so -O0 -shared --section-start cst4=0x800 --section-start str=0x1000
# RUN: llvm-objdump -s %t0.so | FileCheck %s --check-prefix=OBJDUMP0
# RUN: ld.lld %t.o -o %t2.so -O2 -shared --section-start cst4=0x800 --section-start str=0x1000
# RUN: llvm-objdump -s %t2.so | FileCheck %s --check-prefix=OBJDUMP2

# CHECK:       Name              Type            Address          Off    Size   ES Flg Lk Inf Al
# CHECK:       cst4              PROGBITS        0000000000000800 000800 000004 04  AM  0   0  1
# CHECK-NEXT:  str               PROGBITS        0000000000001000 001000 000009 01 AMS  0   0  1

# CHECK:      Relocation section '.rela.dyn'
# CHECK-NEXT:     Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
# CHECK-NEXT:                                    R_X86_64_RELATIVE                         802
# CHECK-NEXT:                                    R_X86_64_RELATIVE                         1000
# CHECK-NEXT:                                    R_X86_64_RELATIVE                         1006
# CHECK-NEXT:                                    R_X86_64_RELATIVE                         1002
# CHECK-EMPTY:

# OBJDUMP:      Contents of section str:
# OBJDUMP-NEXT: 1000 61006162 63006263 00                 a.abc.bc.
# OBJDUMP:      Contents of section .data:
# OBJDUMP-NEXT:       00000000 00000000 00000000 00000000  ................
# OBJDUMP-NEXT:       00000000 00000000                    ........
# OBJDUMP:      Contents of section .bar:
# OBJDUMP-NEXT:  0000 00080000 00000000 00080000 00000000  ................

# OBJDUMP0:      Contents of section cst4:
# OBJDUMP0-NEXT:  0800 2a000000 2a000000                    *...*...
# OBJDUMP0-NEXT: Contents of section str:
# OBJDUMP0-NEXT:  1000 61626300 61006263 00626300           abc.a.bc.bc.

# OBJDUMP2:      Contents of section cst4:
# OBJDUMP2-NEXT:  0800 2a000000                             *...
# OBJDUMP2-NEXT: Contents of section str:
# OBJDUMP2-NEXT:  1000 61626300 6100                        abc.a.

.section cst4,"aM",@progbits,4
.long 42
.long 42

.section str,"aMS",@progbits,1
abc:
.asciz "abc"
a:
.asciz "a"
bc:
.asciz "bc"
.asciz "bc"

.data
.quad cst4 + 6
.quad a
.quad bc
.quad abc

.section .bar
.quad cst4
.quad cst4 + 4
