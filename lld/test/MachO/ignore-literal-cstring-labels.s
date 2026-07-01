# REQUIRES: aarch64

## ld64's Literal{4,8,16}Section and CStringSection return ignoreLabel=true for
## labels whose name starts with 'L' or 'l'. Those labels never become named
## atoms nor enter the symbol table.
##
## Before the fix, LLD ran these labels through SymbolTable::addDefined. An
## 'l'-prefix label on a __literal8 record in one TU could therefore collide
## with the same name on a cstring piece in __objc_methname in another TU,
## producing a spurious `duplicate symbol` diagnostic.

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/p.s -o %t/p.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/n.s -o %t/n.o
# RUN: %lld -arch arm64 -lSystem --icf=all -dylib %t/p.o %t/n.o -o %t/out.dylib
# RUN: llvm-nm --defined-only %t/out.dylib | count 0

#--- p.s
## `l_LIT` on an __objc_methname cstring piece, referenced by a
## __objc_selrefs record.
.subsections_via_symbols

.section __TEXT,__objc_methname,cstring_literals
.private_extern l_LIT
.globl l_LIT
l_LIT:
  .asciz "msg_p"

.section __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align 3
.private_extern l_SEL_P
.globl l_SEL_P
l_SEL_P:
  .quad l_LIT

#--- n.s
## Same `l_LIT` name on a __literal8 record in another TU.
.subsections_via_symbols

.literal8
.p2align 3
.private_extern l_LIT
.globl l_LIT
l_LIT:
  .quad 0x4141414141414141
