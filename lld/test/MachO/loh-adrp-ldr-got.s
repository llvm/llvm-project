# REQUIRES: aarch64

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/obj.s -o %t/obj.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/dylib.s -o %t/dylib.o
# RUN: %lld -arch arm64 -dylib -o %t/libdylib.dylib %t/dylib.o
# RUN: %lld -arch arm64 %t/obj.o %t/libdylib.dylib -o %t/AdrpLdrGot
# RUN: llvm-objdump -d --macho %t/AdrpLdrGot | FileCheck %s

#--- obj.s
.text
.globl _main
# CHECK-LABEL: _main:
_main:
## The referenced symbol is local
L1: adrp x0, _local@GOTPAGE
L2: ldr  x0, [x0, _local@GOTPAGEOFF]
# CHECK-NEXT: adr x0
# CHECK-NEXT: nop

## The referenced symbol is in a dylib
L3: adrp x1, _external@GOTPAGE
L4: ldr  x1, [x1, _external@GOTPAGEOFF]
# CHECK-NEXT: nop
# CHECK-NEXT: ldr x1

_local:
  nop

.loh AdrpLdrGot L1, L2
.loh AdrpLdrGot L3, L4

#--- dylib.s
.globl _external
_external:
