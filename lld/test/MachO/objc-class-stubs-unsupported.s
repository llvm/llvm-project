# REQUIRES: aarch64, x86

# Tests that ObjC class stubs are rejected on architectures without an encoding.

# RUN: rm -rf %t && split-file %s %t

# Check x86_64 rejection.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/x86_64.s \
# RUN:   -o %t/x86_64.o
# RUN: not %lld -arch x86_64 -lSystem -o /dev/null %t/x86_64.o \
# RUN:   -objc_stubs_fast 2>&1 | FileCheck %s --check-prefix=X86_64

# Check arm64_32 rejection.
# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-watchos %t/arm64_32.s \
# RUN:   -o %t/arm64_32.o
# RUN: not %lld-watchos -arch arm64_32 -lSystem -o /dev/null %t/arm64_32.o \
# RUN:   -objc_stubs_fast 2>&1 | FileCheck %s --check-prefix=ARM64_32

# X86_64: error: objc class stubs are not supported for x86_64
# ARM64_32: error: objc class stubs are not supported for arm64_32

#--- x86_64.s
# x86_64 input should reject class-stub symbols.
.text
.globl _main
_main:
  callq _objc_msgSendClass$foo$_OBJC_CLASS_$_Foo
  ret

.data
.globl _OBJC_CLASS_$_Foo
_OBJC_CLASS_$_Foo:
  .quad 0

#--- arm64_32.s
# arm64_32 input should reject class-stub symbols.
.text
.globl _main
_main:
  bl _objc_msgSendClass$foo$_OBJC_CLASS_$_Foo
  ret

.data
.globl _OBJC_CLASS_$_Foo
_OBJC_CLASS_$_Foo:
  .long 0
