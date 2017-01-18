// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: not ld.lld -shared %t.o -o %t 2>&1 | FileCheck %s

// CHECK: error:  Section has flags incompatible with others with the same name {{.*}}incompatible-section-flags.s.tmp.o:(.foo)
// CHECK: error:  Section has flags incompatible with others with the same name {{.*}}incompatible-section-flags.s.tmp.o:(.bar)

.section .foo, "awT", @progbits, unique, 1
.quad 0

.section .foo, "aw", @progbits, unique, 2
.quad 0


.section .bar, "aw", @progbits, unique, 3
.quad 0

.section .bar, "awT", @progbits, unique, 4
.quad 0
