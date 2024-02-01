# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/explicit-selrefs-1.s -o %t/explicit-selrefs-1.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/explicit-selrefs-2.s -o %t/explicit-selrefs-2.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/implicit-selrefs.s -o %t/implicit-selrefs.o

# RUN: %lld -dylib -arch arm64 -lSystem -o %t/explicit-only-no-icf \
# RUN:   %t/explicit-selrefs-1.o %t/explicit-selrefs-2.o -no_fixup_chains
# RUN: llvm-otool -vs __DATA __objc_selrefs %t/explicit-only-no-icf | \
# RUN:   FileCheck %s --check-prefix=EXPLICIT-NO-ICF

## NOTE: ld64 always dedups the selrefs unconditionally, but we only do it when
## ICF is enabled.
# RUN: %lld -dylib -arch arm64 -lSystem -o %t/explicit-only-with-icf \
# RUN:   %t/explicit-selrefs-1.o %t/explicit-selrefs-2.o -no_fixup_chains
# RUN: llvm-otool -vs __DATA __objc_selrefs %t/explicit-only-with-icf \
# RUN:   | FileCheck %s --check-prefix=EXPLICIT-WITH-ICF

# SELREFS: Contents of (__DATA,__objc_selrefs) section
# SELREFS-NEXT: __TEXT:__objc_methname:foo
# SELREFS-NEXT: __TEXT:__objc_methname:bar
# SELREFS-NEXT: __TEXT:__objc_methname:foo
# SELREFS-NEXT: __TEXT:__objc_methname:length
# SELREFS-EMPTY:

## We don't yet support dedup'ing implicitly-defined selrefs.
# RUN: %lld -dylib -arch arm64 -lSystem --icf=all -o %t/explicit-and-implicit \
# RUN:   %t/explicit-selrefs-1.o %t/explicit-selrefs-2.o %t/implicit-selrefs.o \
# RUN:   -no_fixup_chains
# RUN: llvm-otool -vs __DATA __objc_selrefs %t/explicit-and-implicit \
# RUN:   | FileCheck %s --check-prefix=EXPLICIT-AND-IMPLICIT

# EXPLICIT-NO-ICF:       Contents of (__DATA,__objc_selrefs) section
# EXPLICIT-NO-ICF-NEXT:  __TEXT:__objc_methname:foo
# EXPLICIT-NO-ICF-NEXT:  __TEXT:__objc_methname:bar
# EXPLICIT-NO-ICF-NEXT:  __TEXT:__objc_methname:bar
# EXPLICIT-NO-ICF-NEXT:  __TEXT:__objc_methname:foo

# EXPLICIT-WITH-ICF:      Contents of (__DATA,__objc_selrefs) section
# EXPLICIT-WITH-ICF-NEXT: __TEXT:__objc_methname:foo
# EXPLICIT-WITH-ICF-NEXT: __TEXT:__objc_methname:bar

# EXPLICIT-AND-IMPLICIT:      Contents of (__DATA,__objc_selrefs) section
# EXPLICIT-AND-IMPLICIT-NEXT: __TEXT:__objc_methname:foo
# EXPLICIT-AND-IMPLICIT-NEXT: __TEXT:__objc_methname:bar
# NOTE: Ideally this wouldn't exist, but while it does it needs to point to the deduplicated string
# EXPLICIT-AND-IMPLICIT-NEXT: __TEXT:__objc_methname:foo
# EXPLICIT-AND-IMPLICIT-NEXT: __TEXT:__objc_methname:length

#--- explicit-selrefs-1.s
.section  __TEXT,__objc_methname,cstring_literals
lselref1:
  .asciz  "foo"
lselref2:
  .asciz  "bar"

.section  __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align  3
  .quad lselref1
  .quad lselref2
  .quad lselref2

#--- explicit-selrefs-2.s
.section  __TEXT,__objc_methname,cstring_literals
lselref1:
  .asciz  "foo"

.section  __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align  3
  .quad lselref1

#--- implicit-selrefs.s
.text
.globl _objc_msgSend
.p2align 2
_objc_msgSend:
  ret

.p2align 2
_sender:
  bl _objc_msgSend$length
  bl _objc_msgSend$foo
  ret
