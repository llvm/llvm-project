# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-ios13.0 %s -o %t.o
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version ios 14.0 14.0 -no_fixup_chains -objc_relative_method_lists %t.o -o %t.small
# RUN: llvm-readobj --sections %t.small | FileCheck %s --check-prefix=SECTION
# RUN: llvm-objdump --macho --full-contents --section=__objc_methlist %t.small | FileCheck %s --check-prefix=HEADER
# RUN: llvm-objdump --macho --syms %t.small | FileCheck %s --check-prefix=SMALL
# RUN: llvm-objdump --macho --rebase %t.small | FileCheck %s --check-prefix=SMALL-REBASE
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version ios 14.0 14.0 -no_fixup_chains -objc_relative_method_lists -dead_strip %t.o -o %t.dead
# RUN: llvm-readobj --sections %t.dead | FileCheck %s --check-prefix=SECTION
# RUN: llvm-objdump --macho --syms %t.dead | FileCheck %s --check-prefix=SMALL
# RUN: llvm-objdump --macho --rebase %t.dead | FileCheck %s --check-prefix=SMALL-REBASE
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version ios 14.0 14.0 -no_fixup_chains %t.o -o %t.default
# RUN: llvm-objdump --macho --syms %t.default | FileCheck %s --check-prefix=SMALL
# RUN: llvm-objdump --macho --rebase %t.default | FileCheck %s --check-prefix=SMALL-REBASE
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version ios 14.0 14.0 -no_fixup_chains -no_objc_relative_method_lists %t.o -o %t.large
# RUN: llvm-objdump --macho --syms %t.large | FileCheck %s --check-prefix=LARGE
# RUN: llvm-objdump --macho --rebase %t.large | FileCheck %s --check-prefix=LARGE-REBASE
# RUN: %no-arg-lld -dylib -arch arm64 -platform_version ios 13.0 13.0 -no_fixup_chains %t.o -o %t.old
# RUN: llvm-objdump --macho --syms %t.old | FileCheck %s --check-prefix=LARGE
# RUN: llvm-objdump --macho --rebase %t.old | FileCheck %s --check-prefix=LARGE-REBASE
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos %S/objc-category-merging-swift-class-ext.s -o %t.swift.o
# RUN: %lld -arch arm64 -dylib -objc_category_merging -objc_relative_method_lists %t.swift.o -o %t.swift
# RUN: llvm-objdump --macho --objc-meta-data %t.swift | FileCheck %s --check-prefix=SWIFT-MERGE

# Four one-entry lists occupy 4 * (8 + 3 * 4) bytes after conversion.
# SECTION: Name: __objc_methlist
# SECTION: Size: 0x50
# HEADER-COUNT-4: 8000000c
# HEADER-NOT: 8000000c

# SMALL-DAG: __TEXT,__objc_methlist .hidden __INSTANCE_METHODS_C
# SMALL-DAG: __TEXT,__objc_methlist .hidden __CLASS_METHODS_C
# SMALL-DAG: __TEXT,__objc_methlist .hidden __CATEGORY_INSTANCE_METHODS_C
# SMALL-DAG: __TEXT,__objc_methlist .hidden __CATEGORY_CLASS_METHODS_C
# SMALL-DAG: __DATA,__objc_const .hidden __INSTANCE_METHOD_C

# LARGE-DAG: __DATA,__objc_const .hidden __INSTANCE_METHODS_C
# LARGE-DAG: __DATA,__objc_const .hidden __CLASS_METHODS_C
# LARGE-DAG: __DATA,__objc_const .hidden __CATEGORY_INSTANCE_METHODS_C
# LARGE-DAG: __DATA,__objc_const .hidden __CATEGORY_CLASS_METHODS_C
# LARGE-DAG: __DATA,__objc_const .hidden __INSTANCE_METHOD_C

# Five list references, three pointers in the unrecognized list, and one shared selref.
# SMALL-REBASE-COUNT-9: pointer
# SMALL-REBASE-NOT: pointer
# Five list references and three pointers in each of the five absolute lists.
# LARGE-REBASE-COUNT-20: pointer
# LARGE-REBASE-NOT: pointer

# SWIFT-MERGE: baseMethods {{.*}} (struct method_list_t *)
# SWIFT-MERGE-NEXT: entsize 12 (relative)
# SWIFT-MERGE-NEXT: count 1
# SWIFT-MERGE-NEXT: name 0x{{[0-9a-f]+}} (0x{{[0-9a-f]+}}) init
# SWIFT-MERGE-NEXT: types 0x{{[0-9a-f]+}} (0x{{[0-9a-f]+}}) @16@0:8
# SWIFT-MERGE-NEXT: imp 0x{{[0-9a-f]+}} (0x{{[0-9a-f]+}}) _$s11MyTestClassAACABycfcTo
# SWIFT-MERGE-NEXT: baseProtocols 0x0

.text
.globl _imp
_imp:
  ret

.section __TEXT,__objc_methname,cstring_literals
Lselector:
  .asciz "method"
.section __TEXT,__objc_methtype,cstring_literals
Ltypes:
  .asciz "v16@0:8"

.macro method_list name
.section __DATA,__objc_const
.p2align 3
.globl \name
.private_extern \name
\name:
  .long 24
  .long 1
  .quad Lselector
  .quad Ltypes
  .quad _imp
.endm

method_list __INSTANCE_METHODS_C
method_list __CLASS_METHODS_C
method_list __CATEGORY_INSTANCE_METHODS_C
method_list __CATEGORY_CLASS_METHODS_C
method_list __INSTANCE_METHOD_C

.section __DATA,__data
.globl _lists
.p2align 3
_lists:
  .quad __INSTANCE_METHODS_C
  .quad __CLASS_METHODS_C
  .quad __CATEGORY_INSTANCE_METHODS_C
  .quad __CATEGORY_CLASS_METHODS_C
  .quad __INSTANCE_METHOD_C

.subsections_via_symbols
