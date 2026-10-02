# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/pure.s -o %t/pure.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/external.s -o %t/external.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/mixed.s -o %t/mixed.o
# RUN: %lld -arch arm64 -dylib %t/foo.o -o %t/libFoo.dylib -install_name @rpath/libFoo.dylib

## ---------------------------------------------------------------------------
## A section that is nothing but a force-load pointer (anchored by a local
## symbol) is dropped entirely: no bind, and the __const bytes are reclaimed.
## The overlay dylib dependency is preserved.
## ---------------------------------------------------------------------------
# RUN: %lld -arch arm64 -dylib %t/pure.o %t/libFoo.dylib -o %t/pure-noflag.dylib
# RUN: llvm-objdump --macho --bind %t/pure-noflag.dylib | FileCheck %s --check-prefix=PURE-BIND
# RUN: llvm-objdump --macho --section-headers %t/pure-noflag.dylib | FileCheck %s --check-prefix=HAS-CONST

# RUN: %lld -arch arm64 -dylib %t/pure.o %t/libFoo.dylib -o %t/pure-flag.dylib --strip-swift-force-load
# RUN: llvm-objdump --macho --bind %t/pure-flag.dylib | FileCheck %s --check-prefix=NO-FORCE-LOAD
# RUN: llvm-objdump --macho --section-headers %t/pure-flag.dylib | FileCheck %s --check-prefix=NO-CONST
# RUN: llvm-otool -L %t/pure-flag.dylib | FileCheck %s --check-prefix=DYLIB

## --no-strip-swift-force-load overrides an earlier --strip-swift-force-load.
# RUN: %lld -arch arm64 -dylib %t/pure.o %t/libFoo.dylib -o %t/pure-noflag2.dylib --strip-swift-force-load --no-strip-swift-force-load
# RUN: llvm-objdump --macho --bind %t/pure-noflag2.dylib | FileCheck %s --check-prefix=PURE-BIND

## Dropping still happens under -dead_strip (the section is a no_dead_strip root).
# RUN: %lld -arch arm64 -dylib -dead_strip %t/pure.o %t/libFoo.dylib -o %t/pure-ds.dylib --strip-swift-force-load
# RUN: llvm-objdump --macho --bind %t/pure-ds.dylib | FileCheck %s --check-prefix=NO-FORCE-LOAD
# RUN: llvm-otool -L %t/pure-ds.dylib | FileCheck %s --check-prefix=DYLIB

## -dead_strip_dylibs must not drop the section, the imported FORCE_LOAD symbol
## stays referenced even though the section holding its fixup is gone.
# RUN: %lld -arch arm64 -dylib -dead_strip_dylibs %t/pure.o %t/libFoo.dylib -o %t/pure-dsd.dylib --strip-swift-force-load
# RUN: llvm-objdump --macho --bind %t/pure-dsd.dylib | FileCheck %s --check-prefix=NO-FORCE-LOAD
# RUN: llvm-objdump --macho --section-headers %t/pure-dsd.dylib | FileCheck %s --check-prefix=NO-CONST
# RUN: llvm-otool -L %t/pure-dsd.dylib | FileCheck %s --check-prefix=DYLIB

# PURE-BIND:      Bind table:
# PURE-BIND:      __DATA_CONST __const {{.*}} pointer 0 libFoo __swift_FORCE_LOAD_$_swiftFoo
# HAS-CONST:      __const
# NO-FORCE-LOAD-NOT: __swift_FORCE_LOAD_$_swiftFoo
# NO-CONST-NOT:  __const
# DYLIB:         libFoo.dylib

## ---------------------------------------------------------------------------
## A force-load pointer anchored by an externally visible symbol must be kept
## (clients may link against it), so the bind is NOT stripd.
## ---------------------------------------------------------------------------
# RUN: %lld -arch arm64 -dylib %t/external.o %t/libFoo.dylib -o %t/external-flag.dylib --strip-swift-force-load
# RUN: llvm-objdump --macho --bind %t/external-flag.dylib | FileCheck %s --check-prefix=EXT-BIND
# RUN: llvm-nm %t/external-flag.dylib | FileCheck %s --check-prefix=EXT-NM

# EXT-BIND:      __DATA_CONST __const {{.*}} pointer 0 libFoo __swift_FORCE_LOAD_$_swiftFoo
# EXT-NM:        _ext_anchor

## ---------------------------------------------------------------------------
## A section that mixes a force-load pointer with other data is not pure, so it
## is left untouched (no per-reloc fallback): the bind remains.
## ---------------------------------------------------------------------------
# RUN: %lld -arch arm64 -dylib %t/mixed.o %t/libFoo.dylib -o %t/mixed-flag.dylib --strip-swift-force-load
# RUN: llvm-objdump --macho --bind %t/mixed-flag.dylib | FileCheck %s --check-prefix=MIX-BIND

# MIX-BIND:      __DATA_CONST __const {{.*}} pointer 0 libFoo __swift_FORCE_LOAD_$_swiftFoo

#--- foo.s
## Stub overlay dylib exporting the FORCE_LOAD symbol.
.globl __swift_FORCE_LOAD_$_swiftFoo
.data
__swift_FORCE_LOAD_$_swiftFoo:
  .quad 0
.subsections_via_symbols

#--- pure.s
## A __DATA,__const section holding only a force-load pointer, anchored by a
## compiler-local symbol and marked no_dead_strip (as the Swift compiler does).
.section __DATA,__const
.private_extern _pure_anchor
.globl _pure_anchor
.weak_definition _pure_anchor
.p2align 3
_pure_anchor:
  .quad __swift_FORCE_LOAD_$_swiftFoo
.no_dead_strip _pure_anchor
.subsections_via_symbols

#--- external.s
## Same shape, but the anchor is exported -- must be preserved.
.globl _ext_anchor
.section __DATA,__const
.p2align 3
_ext_anchor:
  .quad __swift_FORCE_LOAD_$_swiftFoo
.subsections_via_symbols

#--- mixed.s
## A section mixing a force-load pointer with an unrelated data pointer: not a
## pure force-load section, so it is left untouched.
.section __DATA,__const
.private_extern _mixed_anchor
.globl _mixed_anchor
.weak_definition _mixed_anchor
.p2align 3
_mixed_anchor:
  .quad __swift_FORCE_LOAD_$_swiftFoo
  .quad _some_data
.no_dead_strip _mixed_anchor
.data
_some_data:
  .quad 0
.subsections_via_symbols
