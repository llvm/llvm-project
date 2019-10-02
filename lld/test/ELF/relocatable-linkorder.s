// REQUIRES: x86
// RUN: llvm-mc %s -o %t.o -filetype=obj --triple=x86_64-unknown-linux
// RUN: ld.lld %t.o -o %t -r
// RUN: llvm-readelf -S %t | FileCheck --check-prefix=DIFFERENT %s
// RUN: echo 'SECTIONS { .text.f1 : { *(.text.f1) } .text.f2 : { *(.text.f2) } }' > %t.lds
// RUN: ld.lld %t.o -o %t -r %t.lds
// RUN: llvm-readelf -S %t | FileCheck --check-prefix=DIFFERENT %s
// RUN: echo 'SECTIONS { .text : { *(.text.f1) *(.text.f2) } }' > %t.lds
// RUN: ld.lld %t.o -o %t -r %t.lds
// RUN: llvm-readelf -S -x foo %t | FileCheck --check-prefix=SAME %s

/// Test that SHF_LINK_ORDER sections with different linked sections
/// aren't merged.

.section .text.f1,"ax",@progbits
.globl f1
f1:
ret

.section .text.f2,"ax",@progbits
.globl f2
f2:
ret

// SAME: foo
// DIFFERENT: foo
.section foo,"ao",@progbits,.text.f2,unique,2
.quad 2

// SAME-NOT: foo
// DIFFERENT: foo
.section foo,"ao",@progbits,.text.f1,unique,1
.quad 1

// SAME: Hex dump of section 'foo':
// SAME: 01000000 00000000 02000000 00000000
