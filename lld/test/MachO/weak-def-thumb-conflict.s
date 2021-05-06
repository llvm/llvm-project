# REQUIRES: arm
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=armv7-apple-watchos %t/thumb-foo.s -o %t/thumb-foo.o
# RUN: llvm-mc -filetype=obj -triple=armv7-apple-watchos %t/arm-foo.s -o %t/arm-foo.o
# RUN: %lld-watchos -arch armv7 -dylib %t/arm-foo.o %t/thumb-foo.o -o %t/arm-foo
# RUN: %lld-watchos -arch armv7 -dylib %t/thumb-foo.o %t/arm-foo.o -o %t/thumb-foo
# RUN: llvm-nm -m %t/arm-foo | FileCheck %s --check-prefix=ARM
# RUN: llvm-nm -m %t/thumb-foo | FileCheck %s --check-prefix=THUMB

## Check that we preserve the .thumb_def flag if we pick the thumb definition of
## _foo.
# ARM:   (__TEXT,arm)   weak external         _foo
# THUMB: (__TEXT,thumb) weak external [Thumb] _foo

#--- thumb-foo.s
.section __TEXT,thumb
.globl _foo
.weak_definition _foo
.thumb_func _foo
.p2align 2
_foo:

#--- arm-foo.s
.section __TEXT,arm
.globl _foo
.weak_definition _foo
.p2align 2
_foo:
