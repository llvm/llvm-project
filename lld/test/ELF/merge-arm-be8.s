// REQUIRES: arm
// Ensure that ARM big-endian(BE8) can link SHF_MERGE sections with mapping symbols correctly.

// RUN: llvm-mc -filetype=obj -triple=armv7aeb-none-linux-gnueabi  %s -o %t.o
// RUN: ld.lld --be8 %t.o -o /dev/null

.section .merge, "aM", %progbits, 4
.local $d.1
$d.1:
        .word 4
