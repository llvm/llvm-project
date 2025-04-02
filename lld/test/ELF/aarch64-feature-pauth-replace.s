### This file replace .note.gnu.property with aarch64 build attributes in order to confirm
### interoperability.

# REQUIRES: aarch64

# RUN: rm -rf %t && split-file %s %t && cd %t

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux-gnu abi-tag1.s -o tag1.o
# RUN: cp tag1.o tag1a.o
# RUN: ld.lld -shared tag1.o tag1a.o -o tagok.so
# RUN: llvm-readelf -n tagok.so | FileCheck --check-prefix OK %s

# OK: AArch64 PAuth ABI core info: platform 0x2a (unknown), version 0x1

# RUN: llvm-readelf -d pacplt-nowarn | FileCheck --check-prefix=PACPLTTAG %s
# RUN: llvm-readelf -d pacplt-warn   | FileCheck --check-prefix=PACPLTTAG %s

# PACPLTTAG:      0x0000000070000003 (AARCH64_PAC_PLT)

# RUN: llvm-objdump -d pacplt-nowarn | FileCheck --check-prefixes=PACPLT,NOHINT -DA=10380 -DB=478 -DC=480 %s
# RUN: llvm-objdump -d pacplt-warn   | FileCheck --check-prefixes=PACPLT,HINT   -DA=10390 -DB=488 -DC=490 %s

# PACPLT: Disassembly of section .text:
# PACPLT:      <func2>:
# PACPLT-NEXT:     bl      0x[[A]] <func3@plt>
# PACPLT-NEXT:     ret
# PACPLT: Disassembly of section .plt:
# PACPLT:      <.plt>:
# PACPLT-NEXT:     stp     x16, x30, [sp, #-0x10]!
# PACPLT-NEXT:     adrp    x16, 0x30000 <func3+0x30000>
# PACPLT-NEXT:     ldr     x17, [x16, #0x[[B]]]
# PACPLT-NEXT:     add     x16, x16, #0x[[B]]
# PACPLT-NEXT:     br      x17
# PACPLT-NEXT:     nop
# PACPLT-NEXT:     nop
# PACPLT-NEXT:     nop
# PACPLT:      <func3@plt>:
# PACPLT-NEXT:     adrp    x16, 0x30000 <func3+0x30000>
# PACPLT-NEXT:     ldr     x17, [x16, #0x[[C]]]
# PACPLT-NEXT:     add     x16, x16, #0x[[C]]
# NOHINT-NEXT:     braa    x17, x16
# NOHINT-NEXT:     nop
# HINT-NEXT:       autia1716
# HINT-NEXT:       br      x17
# PACPLT-NEXT:     nop


#--- abi-tag1.s

.aeabi_subsection aeabi_pauthabi, required, uleb128
.aeabi_attribute Tag_PAuth_Platform, 42
.aeabi_attribute Tag_PAuth_Schema, 1


## define _start to avoid missing entry warning and use --fatal-warnings to assert no diagnostic
## allow multiple definitions of _start for simplicity
.weak _start;
_start:
