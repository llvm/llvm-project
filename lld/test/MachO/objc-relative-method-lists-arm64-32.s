# REQUIRES: aarch64
## arm64_32 needs ARM64_RELOC_UNSIGNED and 4-byte pointers for the selector
## references synthesized by relative method-list encoding.
# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-watchos9 %s -o %t.o
# RUN: %no-arg-lld -dylib -arch arm64_32 -platform_version watchos 9 9 \
# RUN:   -objc_relative_method_lists %t.o -o %t.dylib
# RUN: llvm-readobj --sections %t.dylib | FileCheck %s --check-prefix=SECTIONS
# RUN: llvm-objdump --macho --full-contents --section=__objc_methlist %t.dylib | \
# RUN:   FileCheck %s --check-prefix=METHLIST
# RUN: llvm-objdump --macho --rebase %t.dylib | FileCheck %s --check-prefix=REBASE

## Relative records are three 32-bit offsets, the same size as three 32-bit
## pointers, so the list does not shrink.
# SECTIONS:      Name: __objc_methlist
# SECTIONS-NEXT: Segment: __TEXT
# SECTIONS-NEXT: Address:
# SECTIONS-NEXT: Size: 0x14
# SECTIONS:      Name: __objc_selrefs
# SECTIONS-NEXT: Segment: __DATA
# SECTIONS-NEXT: Address:
# SECTIONS-NEXT: Size: 0x4
# METHLIST: 0c000080 01000000
# REBASE: __DATA __objc_selrefs 0x{{[0-9A-F]+}} pointer

.text
.globl _imp
.p2align 2
_imp:
  ret

.section __TEXT,__objc_methname,cstring_literals
Lsel:
  .asciz "method"
.section __TEXT,__objc_methtype,cstring_literals
Ltypes:
  .asciz "v8@0:4"

.section __DATA,__objc_const
.p2align 2
__OBJC_$_INSTANCE_METHODS_Foo:
  .long 12
  .long 1
  .long Lsel
  .long Ltypes
  .long _imp

.section __DATA,__data
.globl _keep
.p2align 2
_keep:
  .long __OBJC_$_INSTANCE_METHODS_Foo

.subsections_via_symbols
