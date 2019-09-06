# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o

## Check that _DYNAMIC symbol is created when creating dynamic output,
## and has hidden visibility and address equal to .dynamic section.
# RUN: ld.lld -shared %t.o -o %t.so
# RUN: llvm-readobj --sections --symbols %t.so | FileCheck %s
# RUN: ld.lld -pie %t.o -o %t
# RUN: llvm-readobj --sections --symbols %t | FileCheck %s

# CHECK:      Section {
# CHECK:        Index: 5
# CHECK:        Name: .dynamic
# CHECK-NEXT:   Type: SHT_DYNAMIC
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:     SHF_WRITE
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x[[ADDR:.*]]
# CHECK:      Symbols [
# CHECK:        Symbol {
# CHECK:          Name: _DYNAMIC
# CHECK-NEXT:     Value: 0x[[ADDR]]
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Binding: Local
# CHECK-NEXT:     Type: None
# CHECK-NEXT:     Other [ (0x2)
# CHECK-NEXT:       STV_HIDDEN
# CHECK-NEXT:     ]
# CHECK-NEXT:     Section: .dynamic
# CHECK-NEXT:   }

# RUN: ld.lld %t.o -o %t2
# RUN: llvm-readobj --sections --symbols %t2 | FileCheck /dev/null --implicit-check-not=_DYNAMIC

.globl _start
_start:
