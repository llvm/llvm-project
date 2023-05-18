# REQUIRES: x86
## In a relocatable link, don't combine SHF_LINK_ORDER and non-SHF_LINK_ORDER
## like we don't combine SHF_LINK_ORDER with different linked-to sections
## (see linkerscript/linkorder-linked-to.s).
## Test we support adding a non-SHF_LINK_ORDER section as an orphan first.

# RUN: llvm-mc -filetype=obj --triple=x86_64 %s -o %t.o

# RUN: ld.lld -r %t.o -o %t.ro
# RUN: llvm-readelf -x foo %t.ro | FileCheck %s

# CHECK:      Hex dump of section 'foo':
# CHECK-NEXT: 0x00000000 0100

.section foo,"a"
.byte 0

.section .text,"ax",@progbits
ret

.section foo,"ao",@progbits,.text
.byte 1
