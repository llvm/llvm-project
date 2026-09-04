# REQUIRES: aarch64

## Compilers currently emit over-broad __llvm_addrsig entries that also cover
## data symbols which aren't actually address-significant. To match ld64 --
## which coalesces __cfstring / __objc_classrefs / __objc_selrefs
## unconditionally -- LLD deliberately disregards addrsig markings on those
## data sections, so the duplicate _cfs2 / _cr2 / _sr2 entries must fold into
## _cfs1 / _cr1 / _sr1 whether the compiler lists data symbols in
## __llvm_addrsig (with-addrsig.o) or emits no __llvm_addrsig at all
## (without-addrsig.o) when ICF is enabled. At --icf=none, only __cfstring
## still folds (via --deduplicate-strings, which is on by default);
## __objc_classrefs and __objc_selrefs stay unique.

# RUN: rm -rf %t && split-file %s %t
# RUN: llvm-mc -I %t -filetype=obj -triple=arm64-apple-darwin %t/with-addrsig.s -o %t/with-addrsig.o
# RUN: llvm-mc -I %t -filetype=obj -triple=arm64-apple-darwin %t/without-addrsig.s -o %t/without-addrsig.o
# RUN: %lld -arch arm64 -lSystem --icf=safe -dylib -map - -o %t/with-addrsig-safe.dylib %t/with-addrsig.o | FileCheck %s --check-prefixes=CHECK,FOLD
# RUN: %lld -arch arm64 -lSystem --icf=safe -dylib -map - -o %t/without-addrsig-safe.dylib %t/without-addrsig.o | FileCheck %s --check-prefixes=CHECK,FOLD
# RUN: %lld -arch arm64 -lSystem --icf=safe_thunks -dylib -map - -o %t/with-addrsig-thunks.dylib %t/with-addrsig.o | FileCheck %s --check-prefixes=CHECK,FOLD
# RUN: %lld -arch arm64 -lSystem --icf=safe_thunks -dylib -map - -o %t/without-addrsig-thunks.dylib %t/without-addrsig.o | FileCheck %s --check-prefixes=CHECK,FOLD
# RUN: %lld -arch arm64 -lSystem --icf=all -dylib -map - -o %t/with-addrsig-all.dylib %t/with-addrsig.o | FileCheck %s --check-prefixes=CHECK,FOLD
# RUN: %lld -arch arm64 -lSystem --icf=all -dylib -map - -o %t/without-addrsig-all.dylib %t/without-addrsig.o | FileCheck %s --check-prefixes=CHECK,FOLD
# RUN: %lld -arch arm64 -lSystem --icf=none -dylib -map - -o %t/with-addrsig-none.dylib %t/with-addrsig.o | FileCheck %s --check-prefixes=CHECK,NOFOLD
# RUN: %lld -arch arm64 -lSystem --icf=none -dylib -map - -o %t/without-addrsig-none.dylib %t/without-addrsig.o | FileCheck %s --check-prefixes=CHECK,NOFOLD

## __cfstring folds whenever --deduplicate-strings is on, i.e. at every ICF
## level including --icf=none.
# CHECK:      0x00000020 [  2] _cfs1
# CHECK-NEXT: 0x00000000 [  2] _cfs2

## __objc_classrefs / __objc_selrefs fold only when ICF runs.
# FOLD:      0x00000008 [  2] _cr1
# FOLD-NEXT: 0x00000000 [  2] _cr2
# FOLD:      0x00000008 [  2] _sr1
# FOLD-NEXT: 0x00000000 [  2] _sr2

# NOFOLD:      0x00000008 [  2] _cr1
# NOFOLD-NEXT: 0x00000008 [  2] _cr2
# NOFOLD:      0x00000008 [  2] _sr1
# NOFOLD-NEXT: 0x00000008 [  2] _sr2

#--- common.s
.subsections_via_symbols

.section __DATA,__cfstring
.p2align 3
.globl _cfs1
_cfs1:
  .quad _class
  .long 1992
  .space 4
  .quad Lstr
  .quad 5
.globl _cfs2
_cfs2:
  .quad _class
  .long 1992
  .space 4
  .quad Lstr
  .quad 5

.section __TEXT,__cstring,cstring_literals
Lstr:
  .asciz "hi"

.section __DATA,__objc_data
.globl _class
_class:
  .quad 42

.section __DATA,__objc_classrefs,regular,no_dead_strip
.p2align 3
.globl _cr1
_cr1:
  .quad _class
.globl _cr2
_cr2:
  .quad _class

.section __TEXT,__objc_methname,cstring_literals
Lsel:
  .asciz "msg"

.section __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align 3
.globl _sr1
_sr1:
  .quad Lsel
.globl _sr2
_sr2:
  .quad Lsel

#--- with-addrsig.s
.include "common.s"

.addrsig
.addrsig_sym _cfs1
.addrsig_sym _cfs2
.addrsig_sym _cr1
.addrsig_sym _cr2
.addrsig_sym _sr1
.addrsig_sym _sr2

#--- without-addrsig.s
.include "common.s"
