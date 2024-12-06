# Regression test for https://github.com/llvm/llvm-project/issues/100096
# static glibc binaries crash on startup because _init is moved and
# shares its address with an array end pointer. The GOT rewriting can't
# tell the two pointers apart and incorrectly updates the _array_end
# address. Test checks that _init is not moved.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -static -Wl,--section-start=.data=0x1000 -Wl,--section-start=.init=0x1004
# RUN: llvm-bolt %t.exe -o %t.bolt
# RUN: llvm-nm %t.exe | FileCheck --check-prefix=CHECK-ORIGINAL %s
# RUN: llvm-nm %t.bolt | FileCheck --check-prefix=CHECK-BOLTED %s

.section .data
.globl _array_end
_array_start:
    .word 0x0

_array_end:
.section .init,"ax",@progbits
.globl _init

# Check that bolt doesn't move _init.
#
# CHECK-ORIGINAL: 0000000000001004 T _init
# CHECK-BOLTED:   0000000000001004 T _init
_init:
    ret

.section .text,"ax",@progbits
.globl _start

# Check that bolt is moving some other functions.
#
# CHECK-ORIGINAL:   0000000000001008 T _start
# CHECK-BOLTED-NOT: 0000000000001008 T _start
_start:
    bl _init
    adrp x0, #:got:_array_end
    ldr x0, [x0, #:gotpage_lo15:_array_end]
    adrp x0, #:got:_init
    ldr x0, [x0, #:gotpage_lo15:_init]
    ret

