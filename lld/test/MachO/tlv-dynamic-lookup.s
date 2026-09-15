# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/consumer.s -o %t/consumer.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/twotlv.s -o %t/twotlv.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/mixed.s -o %t/mixed.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/branch.s -o %t/branch.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/cfi.s -o %t/cfi.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/binder.s -o %t/binder.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/initoff.s -o %t/initoff.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/ep.s -o %t/ep.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/unsmix.s -o %t/unsmix.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/deftlv.s -o %t/deftlv.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/notlv.s -o %t/notlv.o

## A dynamic-lookup symbol's thread-locality is unknown, so its slot goes in
## __got. ld64 lowers this case identically.
# RUN: %lld -arch arm64 -dylib -undefined dynamic_lookup -o %t/dylookup.dylib %t/consumer.o
# RUN: llvm-objdump --macho --section-headers --bind %t/dylookup.dylib | FileCheck %s
# CHECK-NOT:   __thread_ptrs
# CHECK:       __got
# CHECK-LABEL: Bind table:
# CHECK:       __DATA_CONST __got 0x{{[0-9a-f]+}} pointer 0 flat-namespace _tlv

## -U is per-symbol: _tlv is exempt, _other is left undefined.
# RUN: not %lld -arch arm64 -dylib -U _tlv -o /dev/null %t/twotlv.o 2>&1 | \
# RUN:   FileCheck %s --check-prefix=USCOPE
# USCOPE-NOT: _tlv
# USCOPE:     error: undefined symbol: _other
# USCOPE-NOT: _tlv

# RUN: %lld -arch arm64 -dylib -U _tlv -U _other -o %t/u.dylib %t/twotlv.o
# RUN: llvm-objdump --macho --bind %t/u.dylib | FileCheck %s --check-prefix=UBOTH
# UBOTH-DAG: __DATA_CONST __got 0x{{[0-9a-f]+}} pointer 0 flat-namespace _tlv
# UBOTH-DAG: __DATA_CONST __got 0x{{[0-9a-f]+}} pointer 0 flat-namespace _other

# RUN: %no-fatal-warnings-lld -arch arm64 -dylib -flat_namespace -undefined suppress \
# RUN:   -o %t/flat.dylib %t/consumer.o
# RUN: llvm-objdump --macho --bind %t/flat.dylib | FileCheck %s --check-prefix=FLAT
# FLAT: __DATA_CONST __got 0x{{[0-9a-f]+}} pointer 0 flat-namespace _tlv

## Chained fixups must produce a real slot, not just a flat import.
# RUN: %lld -arch arm64 -dylib -undefined dynamic_lookup -fixup_chains \
# RUN:   -o %t/chained.dylib %t/consumer.o
# RUN: llvm-objdump --macho --section-headers --chained-fixups %t/chained.dylib | \
# RUN:   FileCheck %s --check-prefix=CHAINED
# CHAINED-NOT:  __thread_ptrs
# CHAINED:      __got
# CHAINED:      lib_ordinal = -2 (flat-namespace)
# CHAINED:      _tlv

# RUN: %lld -arch arm64 -dylib -install_name @rpath/libnotlv.dylib -o %t/libnotlv.dylib %t/notlv.o
# RUN: %lld -arch arm64 -dylib -install_name @rpath/libtlv.dylib -undefined dynamic_lookup \
# RUN:   -o %t/libtlv.dylib %t/deftlv.o

# RUN: not %lld -arch arm64 -dylib -o /dev/null %t/consumer.o %t/libnotlv.dylib 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR
# ERR: error: {{.*}}TLVP_LOAD_PAGE21 relocation requires that symbol _tlv be thread-local

# RUN: %lld -arch arm64 -dylib -o %t/good.dylib %t/consumer.o %t/libtlv.dylib
# RUN: llvm-objdump --macho --bind %t/good.dylib | FileCheck %s --check-prefix=GOOD
# GOOD: __DATA_CONST __got 0x{{[0-9a-f]+}} pointer 0 libtlv _tlv

## A dynamic-lookup symbol has no thread-locality to publish, so it must not be
## re-exported -- a downstream link would read the missing flag as a definite
## "not thread-local". _readTlv is the positive control: the trie is not simply
## empty for every input.
# RUN: %lld -arch arm64 -dylib -undefined dynamic_lookup -exported_symbol _tlv \
# RUN:   -exported_symbol _readTlv -o %t/reexport.dylib %t/consumer.o
# RUN: llvm-objdump --macho --exports-trie %t/reexport.dylib | \
# RUN:   FileCheck %s --check-prefix=REEXPORT
# REEXPORT:     _readTlv
# REEXPORT-NOT: _tlv{{$}}

## The remaining cases each reach NonLazyPointerSectionBase::addEntry by a route
## that does not pass through prepareSymbolRelocation. None may abort the
## linker: the symbol already has a __got slot by the time the TLV reference
## asks for one, so addEntry must coalesce rather than allocate a second.

## StubsSection::addEntry -> in.got->addEntry under chained fixups, which is the
## default at iOS 16 / macOS 13.
## Chained fixups records binds in LC_DYLD_CHAINED_FIXUPS, so --bind is empty
## here; the stub and the TLV reference share the one __got slot.
# RUN: %no-arg-lld -arch arm64 -platform_version ios 16.0 16.0 -dylib \
# RUN:   -undefined dynamic_lookup -o %t/branch-chained.dylib %t/branch.o
# RUN: llvm-objdump --macho --section-headers --chained-fixups %t/branch-chained.dylib | \
# RUN:   FileCheck %s --check-prefix=BRANCHC
# BRANCHC-NOT: __thread_ptrs
# BRANCHC:     __got
# BRANCHC:     lib_ordinal = -2 (flat-namespace)

## Same source without chained fixups must not reach a different verdict.
# RUN: %lld -arch arm64 -dylib -undefined dynamic_lookup -no_fixup_chains \
# RUN:   -o %t/branch-lazy.dylib %t/branch.o
# RUN: llvm-objdump --macho --bind %t/branch-lazy.dylib | FileCheck %s --check-prefix=BRANCH
# BRANCH: __DATA_CONST __got 0x{{[0-9a-f]+}} pointer 0 flat-namespace _tlv

## UnwindInfoSection::prepare, after the relocation scan.
# RUN: %lld -arch arm64 -dylib -undefined dynamic_lookup -o %t/cfi.dylib %t/cfi.o
# RUN: llvm-objdump --macho --bind %t/cfi.dylib | FileCheck %s --check-prefix=CFI
# CFI: __DATA_CONST __got 0x{{[0-9a-f]+}} pointer 0 flat-namespace _pers

## StubHelperSection::setUp, which runs after Writer::run's errorCount() bail
## and so can never be protected by an error() emitted earlier.
# RUN: %no-lsystem-lld -arch arm64 -dylib -undefined dynamic_lookup -no_fixup_chains \
# RUN:   -o %t/binder.dylib %t/binder.o
# RUN: llvm-objdump --macho --bind %t/binder.dylib | FileCheck %s --check-prefix=BINDER
# BINDER: __DATA_CONST __got 0x{{[0-9a-f]+}} pointer 0 flat-namespace dyld_stub_binder

## InitOffsetsSection::setUp.
# RUN: %no-arg-lld -arch arm64 -platform_version macos 13.0 13.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem -dylib -undefined dynamic_lookup \
# RUN:   -init_offsets -o %t/initoff.dylib %t/initoff.o
# RUN: llvm-objdump --macho --section-headers --chained-fixups %t/initoff.dylib | \
# RUN:   FileCheck %s --check-prefix=INITOFF
# INITOFF-NOT: __thread_ptrs
# INITOFF:     __got
# INITOFF:     lib_ordinal = -2 (flat-namespace)
# INITOFF:     _ctor

## The entry-point stub is synthesized before the relocation scan; the only
## reference in the source is the TLV one, so there is nothing for a user to fix.
# RUN: %no-arg-lld -arch arm64 -platform_version macos 13.0 13.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem -e _ep -undefined dynamic_lookup \
# RUN:   -o %t/ep.out %t/ep.o
# RUN: llvm-objdump --macho --section-headers --chained-fixups %t/ep.out | \
# RUN:   FileCheck %s --check-prefix=EP
# EP-NOT: __thread_ptrs
# EP:     __got
# EP:     lib_ordinal = -2 (flat-namespace)
# EP:     _ep

## JUDGMENT CALL. A data pointer and a TLV reference to one dynamic-lookup
## symbol cannot both describe whatever dyld finds, but neither can lld tell
## which is wrong, and the UNSIGNED path allocates no slot for a conflict check
## to inspect. Accepting both binds matches how every other unverifiable
## dynamic-lookup mismatch is already treated.
# RUN: %lld -arch arm64 -dylib -undefined dynamic_lookup -o %t/unsmix.dylib %t/unsmix.o
# RUN: llvm-objdump --macho --bind %t/unsmix.dylib | FileCheck %s --check-prefix=UNS
# UNS-DAG: __DATA __data 0x{{[0-9a-f]+}} pointer 0 flat-namespace _tlv
# UNS-DAG: __DATA_CONST __got 0x{{[0-9a-f]+}} pointer 0 flat-namespace _tlv

## Referencing one symbol both ways yields a single slot, as ld64 emits here.
# RUN: %lld -arch arm64 -dylib -undefined dynamic_lookup -o %t/mixed.dylib %t/mixed.o
# RUN: llvm-objdump --macho --section-headers --bind %t/mixed.dylib | \
# RUN:   FileCheck %s --check-prefix=MIXED
# MIXED-NOT:   __thread_ptrs
# MIXED:       __got
# MIXED-LABEL: Bind table:
# MIXED:       __DATA_CONST __got 0x{{[0-9a-f]+}} pointer 0 flat-namespace _tlv
# MIXED-NOT:   _tlv

#--- consumer.s
.globl _readTlv
.p2align 2
_readTlv:
  adrp x8, _tlv@TLVPPAGE
  ldr  x8, [x8, _tlv@TLVPPAGEOFF]
  ret
.subsections_via_symbols

#--- twotlv.s
.globl _two
.p2align 2
_two:
  adrp x8, _tlv@TLVPPAGE
  ldr  x8, [x8, _tlv@TLVPPAGEOFF]
  adrp x9, _other@TLVPPAGE
  ldr  x9, [x9, _other@TLVPPAGEOFF]
  ret
.subsections_via_symbols

#--- mixed.s
.globl _viaTlv, _viaGot
.p2align 2
_viaTlv:
  adrp x8, _tlv@TLVPPAGE
  ldr  x8, [x8, _tlv@TLVPPAGEOFF]
  ret
_viaGot:
  adrp x8, _tlv@GOTPAGE
  ldr  x8, [x8, _tlv@GOTPAGEOFF]
  ret
.subsections_via_symbols

#--- branch.s
.globl _g
.p2align 2
_g:
  bl _tlv
  adrp x8, _tlv@TLVPPAGE
  ldr  x8, [x8, _tlv@TLVPPAGEOFF]
  ret
.subsections_via_symbols

#--- cfi.s
.globl _f
.p2align 2
_f:
  .cfi_startproc
  .cfi_personality 155, _pers
  adrp x8, _pers@TLVPPAGE
  ldr  x8, [x8, _pers@TLVPPAGEOFF]
  ret
  .cfi_endproc
.subsections_via_symbols

#--- binder.s
.globl _f
.p2align 2
_f:
  adrp x8, dyld_stub_binder@TLVPPAGE
  ldr  x8, [x8, dyld_stub_binder@TLVPPAGEOFF]
  bl _lazy
  ret
.subsections_via_symbols

#--- initoff.s
.globl _f
.p2align 2
_f:
  adrp x8, _ctor@TLVPPAGE
  ldr  x8, [x8, _ctor@TLVPPAGEOFF]
  ret
.section __DATA,__mod_init_func,mod_init_funcs
.quad _ctor
.subsections_via_symbols

#--- ep.s
.globl _main
.p2align 2
_main:
  adrp x8, _ep@TLVPPAGE
  ldr  x8, [x8, _ep@TLVPPAGEOFF]
  ret
.subsections_via_symbols

#--- unsmix.s
.globl _a
.p2align 2
_a:
  adrp x8, _tlv@TLVPPAGE
  ldr  x8, [x8, _tlv@TLVPPAGEOFF]
  ret
.data
.globl _p
_p:
  .quad _tlv
.subsections_via_symbols

#--- deftlv.s
.globl _tlv
.section __DATA,__thread_data,thread_local_regular
_tlv$tlv$init:
  .quad 0
.section __DATA,__thread_vars,thread_local_variables
_tlv:
  .quad __tlv_bootstrap
  .quad 0
  .quad _tlv$tlv$init

#--- notlv.s
.globl _tlv
.data
_tlv:
  .quad 0
