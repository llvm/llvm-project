// REQUIRES: mips
// RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux %s -o %t
// RUN: ld.lld -shared %t -o %t2
// RUN: llvm-readelf --sections --symbols %t2 | FileCheck %s

// The Mips _GLOBAL_OFFSET_TABLE_ should be defined at the start of the .got

.globl  a
.hidden a
.type   a,@object
.comm   a,4,4

.globl  f
.type   f,@function
f:
 ld      $v0,%got_page(a)($gp)
 daddiu  $v0,$v0,%got_ofst(a)

.global _start
.type _start,@function
_start:
 lw      $t0,%call16(f)($gp)
 .word _GLOBAL_OFFSET_TABLE_ - .

// CHECK: {{.*}} .got PROGBITS [[GOT:[0-9a-f]+]]
// CHECK: {{.*}} [[GOT]] {{.*}} _GLOBAL_OFFSET_TABLE_
