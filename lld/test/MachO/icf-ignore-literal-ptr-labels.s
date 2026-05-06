# REQUIRES: aarch64

## ld64's PointerToCStringSection / ObjC2ClassRefsSection / CFStringSection
## have `ignoreLabel=true`. Labels on records in these sections never become
## named atoms. When LLD treated such labels as external symbols, an
## identically-named symbol in another section (e.g. a cstring in
## __objc_methname) would trigger the `replaceSymbol` path in
## SymbolTable::addDefined. That silently replaced the Defined in-place with
## the other-section's state while leaving stale pointers in the record-split
## subsection's symbols list. ICF::foldIdentical then asserted
## `(*it)->value == 0` because the stale Defineds carried the other section's
## offsets.
##
## This test exercises exactly that shape: labels `l005` / `l007` appear both
## on an __objc_selrefs record and on cstring pieces in another TU.

# RUN: rm -rf %t && split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/m.s  -o %t/m.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/o.s  -o %t/o.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/e1.s -o %t/e1.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/e2.s -o %t/e2.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/e3.s -o %t/e3.o

# RUN: %lld -arch arm64 -lSystem --icf=all -dylib %t/m.o %t/e1.o %t/e2.o %t/e3.o %t/o.o -o %t/out.dylib
# RUN: llvm-nm --defined-only %t/out.dylib | count 0

#--- m.s
## Master: `l005` / `l007` at the SAME offset on a single __objc_selrefs record
.subsections_via_symbols

.section __TEXT,__objc_methname,cstring_literals
l001:
  .asciz "msg"

.section __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align 3
.private_extern l005
.globl l005
.private_extern l007
.globl l007
l005:
l007:
  .quad l001

#--- o.s
## Override: same `l005` / `l007` names, this time on cstring pieces in
## __objc_methname.

.section __TEXT,__objc_methname,cstring_literals
lpre:
  .asciz "padding0123456789012345678901234567890123456789012345"
.private_extern l005
.globl l005
l005:
  .asciz "content_thirty_seven_byte_string__xx"
.private_extern l007
.globl l007
l007:
  .asciz "abc"

#--- e1.s
## Extras with unique label names and identical selrefs content, so ICF
## forms a fold group that includes m.o's now-coalesced selrefs subsection.
.subsections_via_symbols
.section __TEXT,__objc_methname,cstring_literals
lx1:
  .asciz "msg"
.section __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align 3
.private_extern lE1
.globl lE1
lE1:
  .quad lx1

#--- e2.s
.subsections_via_symbols
.section __TEXT,__objc_methname,cstring_literals
lx2:
  .asciz "msg"
.section __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align 3
.private_extern lE2
.globl lE2
lE2:
  .quad lx2

#--- e3.s
.subsections_via_symbols
.section __TEXT,__objc_methname,cstring_literals
lx3:
  .asciz "msg"
.section __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align 3
.private_extern lE3
.globl lE3
lE3:
  .quad lx3
