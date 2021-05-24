// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/comdat.s -o %t2.o
// RUN: ld.lld -shared %t.o %t2.o -o %t
// RUN: llvm-objdump -d %t | FileCheck %s
// RUN: llvm-readelf -S -s %t | FileCheck --check-prefix=READ %s

// Check that we don't crash with --gc-section and that we print a list of
// reclaimed sections on stderr.
// RUN: ld.lld --gc-sections --print-gc-sections -shared %t.o %t.o %t2.o -o %t \
// RUN:   2>&1 | FileCheck --check-prefix=GC %s
// GC: removing unused section {{.*}}.o:(.text)
// GC: removing unused section {{.*}}.o:(.text3)
// GC: removing unused section {{.*}}.o:(.text)
// GC: removing unused section {{.*}}.o:(.text)

.globl foo
        .section	.text2,"axG",@progbits,foo,comdat,unique,0
foo:
        nop

// CHECK: Disassembly of section .text2:
// CHECK-EMPTY:
// CHECK-NEXT: <foo>:
// CHECK-NEXT:   nop
// CHECK-NOT: nop

        .section bar, "ax"
        call foo

// CHECK: Disassembly of section bar:
// CHECK-EMPTY:
// CHECK-NEXT: <bar>:
// CHECK-NEXT:   callq  {{.*}} <foo@plt>

.weak zed
zed:
        .section .text3,"axG",@progbits,zed,comdat,unique,0

# READ: .text2 PROGBITS {{.*}} AX
# READ: .text3 PROGBITS {{.*}} AX

# SYM:  NOTYPE LOCAL  DEFAULT UND
# SYM:  NOTYPE LOCAL  HIDDEN  [[#]] _DYNAMIC
# SYM:  NOTYPE GLOBAL DEFAULT [[#]] foo
# SYM:  NOTYPE GLOBAL DEFAULT [[#]] zed
# SYM:  NOTYPE GLOBAL DEFAULT UND   abc
