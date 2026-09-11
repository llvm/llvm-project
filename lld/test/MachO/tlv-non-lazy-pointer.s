# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/libtlv.s -o %t/libtlv.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/tlv.s -o %t/tlv.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/got.s -o %t/got.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/mixed.s -o %t/mixed.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/branch.s -o %t/branch.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/cfi.s -o %t/cfi.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/direct.s -o %t/direct.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/unsigned.s -o %t/unsigned.o

# RUN: %lld -arch arm64 -dylib -lSystem -install_name @rpath/libtlv.dylib \
# RUN:   -o %t/libtlv.dylib %t/libtlv.o

## ld-prime puts an imported TLV's descriptor pointer in __got, even when the
## input only has TLV relocations.
# RUN: %lld -arch arm64 -dylib -o %t/tlv.dylib %t/tlv.o %t/libtlv.dylib
# RUN: llvm-objdump --macho --section-headers --bind %t/tlv.dylib | \
# RUN:   FileCheck %s --check-prefix=TLV
# TLV-NOT:   __thread_ptrs
# TLV:       __got
# TLV-LABEL: Bind table:
# TLV:       __DATA_CONST __got 0x{{[0-9a-f]+}} pointer 0 libtlv _foo

## An ordinary GOT reference asks for the same descriptor address.
# RUN: %lld -arch arm64 -dylib -o %t/got.dylib %t/got.o %t/libtlv.dylib
# RUN: llvm-objdump --macho --section-headers --bind %t/got.dylib | \
# RUN:   FileCheck %s --check-prefix=GOT
# GOT-NOT:   __thread_ptrs
# GOT:       __got
# GOT-LABEL: Bind table:
# GOT:       __DATA_CONST __got 0x{{[0-9a-f]+}} pointer 0 libtlv _foo

## GOT and TLV references must coalesce to one slot, independent of input
## relocation kind.
# RUN: %lld -arch arm64 -dylib -o %t/mixed.dylib %t/mixed.o %t/libtlv.dylib
# RUN: llvm-objdump --macho --section-headers --bind %t/mixed.dylib | \
# RUN:   FileCheck %s --check-prefix=MIXED
# MIXED-NOT:   __thread_ptrs
# MIXED:       __got
# MIXED-LABEL: Bind table:
# MIXED:       __DATA_CONST __got 0x{{[0-9a-f]+}} pointer 0 libtlv _foo
# MIXED-NOT:   _foo

## A stub needs a GOT entry under chained fixups. It and the TLV relocation
## must share that entry rather than asserting while allocating two slots.
# RUN: %lld -arch arm64 -dylib -fixup_chains -o %t/branch.dylib \
# RUN:   %t/branch.o %t/libtlv.dylib
# RUN: llvm-objdump --macho --section-headers --chained-fixups \
# RUN:   %t/branch.dylib | FileCheck %s --check-prefix=BRANCH
# BRANCH-NOT: __thread_ptrs
# BRANCH:     __got
# BRANCH:     _foo
# BRANCH-NOT: _foo

## Compact unwind personalities also request a GOT entry after the ordinary
## relocation scan. That late request must find the same slot.
# RUN: %lld -arch arm64 -dylib -o %t/cfi.dylib %t/cfi.o %t/libtlv.dylib
# RUN: llvm-objdump --macho --section-headers --bind %t/cfi.dylib | \
# RUN:   FileCheck %s --check-prefix=CFI
# CFI-NOT:   __thread_ptrs
# CFI:       __got
# CFI-LABEL: Bind table:
# CFI:       __DATA_CONST __got 0x{{[0-9a-f]+}} pointer 0 libtlv _foo
# CFI-NOT:   _foo

## An unsigned relocation is a request for the descriptor's address. It binds
## independently of the TLV reference's non-lazy pointer, as ld-prime does.
# RUN: %lld -arch arm64 -dylib -o %t/unsigned.dylib %t/unsigned.o \
# RUN:   %t/libtlv.dylib
# RUN: llvm-objdump --macho --bind %t/unsigned.dylib | \
# RUN:   FileCheck %s --check-prefix=UNSIGNED
# UNSIGNED-DAG: __DATA_CONST __got  0x{{[0-9a-f]+}} pointer 0 libtlv _foo
# UNSIGNED-DAG: __DATA       __data 0x{{[0-9a-f]+}} pointer 0 libtlv _foo

## A direct address relocation is valid for a descriptor defined in the same
## image, but an imported descriptor has no fixed link-time address. ld-prime
## accepts the former and rejects the latter.
# RUN: %lld -arch arm64 -dylib -lSystem -o %t/direct-local.dylib \
# RUN:   %t/direct.o %t/libtlv.o
# RUN: not %lld -arch arm64 -dylib -o /dev/null %t/direct.o \
# RUN:   %t/libtlv.dylib 2>&1 | FileCheck %s --check-prefix=DIRECT
# DIRECT: PAGE21 relocation cannot reference imported thread-local symbol _foo; its TLV descriptor has no address at link time

#--- libtlv.s
.section __DATA,__thread_data,thread_local_regular
_foo$tlv$init:
  .quad 0
.section __DATA,__thread_vars,thread_local_variables
.globl _foo
_foo:
  .quad __tlv_bootstrap
  .quad 0
  .quad _foo$tlv$init

#--- tlv.s
.globl _viaTlv
.p2align 2
_viaTlv:
  adrp x8, _foo@TLVPPAGE
  ldr  x8, [x8, _foo@TLVPPAGEOFF]
  ret
.subsections_via_symbols

#--- got.s
.globl _viaGot
.p2align 2
_viaGot:
  adrp x8, _foo@GOTPAGE
  ldr  x8, [x8, _foo@GOTPAGEOFF]
  ret
.subsections_via_symbols

#--- mixed.s
.globl _viaTlv, _viaGot
.p2align 2
_viaTlv:
  adrp x8, _foo@TLVPPAGE
  ldr  x8, [x8, _foo@TLVPPAGEOFF]
  ret
_viaGot:
  adrp x8, _foo@GOTPAGE
  ldr  x8, [x8, _foo@GOTPAGEOFF]
  ret
.subsections_via_symbols

#--- branch.s
.globl _viaBranchAndTlv
.p2align 2
_viaBranchAndTlv:
  bl _foo
  adrp x8, _foo@TLVPPAGE
  ldr  x8, [x8, _foo@TLVPPAGEOFF]
  ret
.subsections_via_symbols

#--- cfi.s
.globl _viaCfiAndTlv
.p2align 2
_viaCfiAndTlv:
  .cfi_startproc
  .cfi_personality 155, _foo
  adrp x8, _foo@TLVPPAGE
  ldr  x8, [x8, _foo@TLVPPAGEOFF]
  ret
  .cfi_endproc
.subsections_via_symbols

#--- direct.s
.globl _direct
.p2align 2
_direct:
  adrp x8, _foo@PAGE
  add  x8, x8, _foo@PAGEOFF
  ret
.subsections_via_symbols

#--- unsigned.s
.globl _viaTlv
.p2align 2
_viaTlv:
  adrp x8, _foo@TLVPPAGE
  ldr  x8, [x8, _foo@TLVPPAGEOFF]
  ret
.data
.globl _descriptor
_descriptor:
  .quad _foo
.subsections_via_symbols
