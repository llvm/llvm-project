# REQUIRES: x86, zlib
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld -r -o %t %t.o %t.o
# RUN: llvm-readelf -r -x .data -x .bar -x .debug_line %t | FileCheck --check-prefix=RELA %s

# RELA:         Offset          Info         Type               Symbol's Value  Symbol's Name + Addend
# RELA-NEXT:  0000000000000000  {{.*}} R_X86_64_32            0000000000000000 .text + 1
# RELA-NEXT:  0000000000000004  {{.*}} R_X86_64_32            0000000000000000 .text + 5
# RELA-EMPTY:
# RELA:         Offset          Info         Type               Symbol's Value  Symbol's Name + Addend
# RELA-NEXT:  0000000000000000  {{.*}} R_X86_64_64            0000000000000000 .foo + 1
# RELA-NEXT:  0000000000000008  {{.*}} R_X86_64_32            0000000000000000 .text + 0
# RELA-NEXT:  000000000000000c  {{.*}} R_X86_64_64            0000000000000000 .foo + 2
# RELA-NEXT:  0000000000000014  {{.*}} R_X86_64_32            0000000000000000 .text + 4
# RELA-EMPTY:
# RELA:         Offset          Info         Type               Symbol's Value  Symbol's Name + Addend
# RELA-NEXT:  0000000000000000  {{.*}} R_X86_64_64            0000000000000000 .foo + 1
# RELA-NEXT:  0000000000000008  {{.*}} R_X86_64_32            0000000000000000 .text + 0
# RELA-NEXT:  000000000000000c  {{.*}} R_X86_64_64            0000000000000000 .foo + 2
# RELA-NEXT:  0000000000000014  {{.*}} R_X86_64_32            0000000000000000 .text + 4

# RELA:       Hex dump of section '.data':
# RELA-NEXT:  0x00000000 00000000 00000000                   ........
# RELA:       Hex dump of section '.bar':
# RELA-NEXT:  0x00000000 00000000 00000000 00000000 00000000 ................
# RELA-NEXT:  0x00000010 00000000 00000000                   ........
# RELA:       Hex dump of section '.debug_line':
# RELA-NEXT:  0x00000000 00000000 00000000 00000000 00000000 ................
# RELA-NEXT:  0x00000010 00000000 00000000                   ........

# RUN: llvm-mc -filetype=obj -triple=i686 %s -o %t1.o
# RUN: ld.lld -r -o %t1 %t1.o %t1.o
# RUN: llvm-readelf -r -x .data -x .bar -x .debug_line %t1 | FileCheck %s --check-prefixes=REL,REL0
# RUN: ld.lld -r --compress-debug-sections=zlib -o %t1.zlib %t1.o %t1.o
# RUN: llvm-objcopy --decompress-debug-sections %t1.zlib %t1.zlib.de
# RUN: llvm-readelf -r -x .data -x .bar -x .debug_line %t1.zlib.de | FileCheck %s --check-prefixes=REL,REL1

# REL:         Offset   Info   Type                Sym. Value  Symbol's Name
# REL-NEXT:  00000000  {{.*}} R_386_32               00000000   .text
# REL-NEXT:  00000004  {{.*}} R_386_32               00000000   .text
# REL-EMPTY:
# REL:        Offset    Info   Type                Sym. Value  Symbol's Name
# REL-NEXT:  00000000  {{.*}} R_386_32               00000000   .foo
# REL-NEXT:  00000004  {{.*}} R_386_32               00000000   .text
# REL-NEXT:  00000008  {{.*}} R_386_32               00000000   .foo
# REL-NEXT:  0000000c  {{.*}} R_386_32               00000000   .text
# REL-EMPTY:
# REL:         Offset   Info   Type                Sym. Value  Symbol's Name
# REL-NEXT:  00000000  {{.*}} R_386_32               00000000   .foo
# REL-NEXT:  00000004  {{.*}} R_386_32               00000000   .text
# REL-NEXT:  00000008  {{.*}} R_386_32               00000000   .foo
# REL-NEXT:  0000000c  {{.*}} R_386_32               00000000   .text

# REL:       Hex dump of section '.data':
# REL-NEXT:  0x00000000 01000000 05000000                   ........
# REL:       Hex dump of section '.bar':
# REL-NEXT:  0x00000000 01000000 00000000 02000000 04000000 ................
# REL0:      Hex dump of section '.debug_line':
# REL0-NEXT: 0x00000000 01000000 00000000 02000000 04000000 ................
## FIXME: https://github.com/llvm/llvm-project/issues/66738 The implicit addends for the second input section are wrong.
# REL1:      Hex dump of section '.debug_line':
# REL1-NEXT: 0x00000000 01000000 00000000 01000000 00000000 ................

.long 42
.data
.long .text + 1

.section .foo
.byte 0

.section .bar
.dc.a .foo + 1
.dc.l .text

.section .debug_line
.dc.a .foo + 1
.dc.l .text
