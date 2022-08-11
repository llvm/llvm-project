# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/existing.s -o %t/existing.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/main.s -o %t/main.o

# RUN: %lld -arch arm64 -lSystem -o %t/out %t/existing.o %t/main.o
# RUN: llvm-otool -vs __DATA __objc_selrefs %t/out | FileCheck %s --check-prefix=SELREFS

# SELREFS: Contents of (__DATA,__objc_selrefs) section
# SELREFS-NEXT: __TEXT:__objc_methname:foo
# SELREFS-NEXT: __TEXT:__objc_methname:bar
# SELREFS-NEXT: __TEXT:__objc_methname:foo
# SELREFS-NEXT: __TEXT:__objc_methname:length
# SELREFS-EMPTY:

# RUN: %lld -arch arm64 -lSystem -o %t/out %t/existing.o %t/main.o --deduplicate-literals
# RUN: llvm-otool -vs __DATA __objc_selrefs %t/out | FileCheck %s --check-prefix=DEDUP

# DEDUP: Contents of (__DATA,__objc_selrefs) section
# DEDUP-NEXT: __TEXT:__objc_methname:foo
# DEDUP-NEXT: __TEXT:__objc_methname:bar
# NOTE: Ideally this wouldn't exist, but while it does it needs to point to the deduplicated string
# DEDUP-NEXT: __TEXT:__objc_methname:foo
# DEDUP-NEXT: __TEXT:__objc_methname:length
# DEDUP-EMPTY:

#--- existing.s
.section  __TEXT,__objc_methname,cstring_literals
lselref1:
  .asciz  "foo"
lselref2:
  .asciz  "bar"

.section  __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align  3
.quad lselref1
.quad lselref2

#--- main.s
.text
.globl _objc_msgSend
_objc_msgSend:
  ret

.globl _main
_main:
  bl  _objc_msgSend$length
  bl  _objc_msgSend$foo
  ret
