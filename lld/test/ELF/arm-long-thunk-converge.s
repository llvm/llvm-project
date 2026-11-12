// REQUIRES: arm
// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: llvm-mc -triple armv7-unknown-gnu -arm-add-build-attributes -filetype=obj -o a.o a.s
// RUN: ld.lld a.o a.lds -o exe
// RUN: llvm-objdump --no-print-imm-hex -d --start-address=0x00000000 --stop-address=0x00000010 --triple=armv7a-linux-gnueabihf exe | FileCheck --check-prefix=CHECK1 %s
// RUN: llvm-objdump --no-print-imm-hex -d --start-address=0x02000000 --stop-address=0x02000010 --triple=armv7a-linux-gnueabihf exe | FileCheck --check-prefix=CHECK2 %s
// RUN: rm -f a.o exe

//--- a.lds
SECTIONS {
  .foo 0x0 : AT(0x0) { *(.foo) }
  .bar 0x2000000 : AT(0x2000000) { *(.bar) }
}

//--- a.s

// CHECK1: <__ARMv7ABSLongThunk_bar>:
// CHECK1-NEXT:        0:       e300c00c        movw    r12, #12
// CHECK1-NEXT:        4:       e340c200        movt    r12, #512
// CHECK1-NEXT:        8:       e12fff1c        bx      r12
// CHECK1: <foo>:
// CHECK1-NEXT:        c:       ebfffffb        bl      0x0 <__ARMv7ABSLongThunk_bar>

.section .foo,"ax",%progbits,unique,1
foo:
bl bar

// CHECK2: <__ARMv7ABSLongThunk_foo>:
// CHECK2-NEXT:  2000000:       e300c00c        movw    r12, #12
// CHECK2-NEXT:  2000004:       e340c000        movt    r12, #0
// CHECK2-NEXT:  2000008:       e12fff1c        bx      r12
// CHECK2: <bar>:
// CHECK2-NEXT:  200000c:       ebfffffb        bl      0x2000000 <__ARMv7ABSLongThunk_foo>

.section .bar,"ax",%progbits,unique,1
bar:
bl foo
.zero 0x1000000
