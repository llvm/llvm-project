# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: %lld -arch arm64 -lSystem  -fixup_chains -o %t.out %t.o
# RUN: llvm-otool -vs __TEXT __objc_stubs %t.out | FileCheck %s --check-prefix=CHECK --check-prefix=FIRST

# Prepend a dummy entry to check if the address for _objc_msgSend is valid.
# RUN: %lld -arch arm64 -lSystem  -fixup_chains -e _dummy -U _dummy -o %t.out %t.o
# RUN: llvm-otool -vs __TEXT __objc_stubs %t.out | FileCheck %s --check-prefix=CHECK --check-prefix=SECOND

# CHECK: Contents of (__TEXT,__objc_stubs) section

# CHECK-NEXT: _objc_msgSend$foo:
# CHECK-NEXT: adrp    x1, 8 ; 0x100008000
# CHECK-NEXT: ldr     x1, [x1]
# CHECK-NEXT: adrp    x16, 4 ; 0x100004000
# FIRST-NEXT: ldr     x16, [x16]
# SECOND-NEXT:ldr     x16, [x16, #0x8]
# CHECK-NEXT: br      x16
# CHECK-NEXT: brk     #0x1
# CHECK-NEXT: brk     #0x1
# CHECK-NEXT: brk     #0x1

# CHECK-EMPTY:

.text
.globl _objc_msgSend
_objc_msgSend:
  ret

.globl _main
_main:
  bl  _objc_msgSend$foo
  ret
