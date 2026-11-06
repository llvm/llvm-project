# REQUIRES: aarch64

# Tests arm64 ObjC class stub emission for fast/small stubs, local/dylib/archive
# classes, autolinked archives, dynamic lookup, malformed names, and missing
# classes.

# RUN: rm -rf %t && split-file %s %t

# Check fast and small stubs for regular selector calls, local class calls, and
# dylib class calls.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/external.s -o %t/external.o
# RUN: %lld -arch arm64 -dylib -install_name @executable_path/libexternal.dylib \
# RUN:   -o %t/libexternal.dylib %t/external.o
# RUN: %lld -arch arm64 -lSystem -o %t/fast.out %t/main.o \
# RUN:   %t/libexternal.dylib -objc_stubs_fast
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs --syms \
# RUN:   --macho %t/fast.out | FileCheck %s --check-prefix=FAST
# RUN: %lld -arch arm64 -lSystem -o %t/small.out %t/main.o \
# RUN:   %t/libexternal.dylib -objc_stubs_small
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs --syms \
# RUN:   --macho %t/small.out | FileCheck %s --check-prefix=SMALL

# Check arm64e. Fast stubs reach _objc_msgSend through __auth_got and braa;
# small stubs branch to it or to its stub, as on arm64.
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-darwin %t/main.s -o %t/main-arm64e.o
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-darwin %t/external.s -o %t/external-arm64e.o
# RUN: %lld -arch arm64e -dylib -install_name @executable_path/libexternal.dylib \
# RUN:   -o %t/libexternal-arm64e.dylib %t/external-arm64e.o
# RUN: %lld -arch arm64e -lSystem -o %t/fast-arm64e.out %t/main-arm64e.o \
# RUN:   %t/libexternal-arm64e.dylib -objc_stubs_fast
# RUN: llvm-objdump --no-show-raw-insn --section=__objc_stubs --syms \
# RUN:   --section=__auth_got --section-headers --macho %t/fast-arm64e.out \
# RUN:   | FileCheck %s --check-prefix=FAST-ARM64E
# RUN: llvm-objdump --macho --dyld-info %t/fast-arm64e.out \
# RUN:   | FileCheck %s --check-prefix=FAST-ARM64E-GOT
# RUN: %lld -arch arm64e -lSystem -o %t/small-arm64e.out %t/main-arm64e.o \
# RUN:   %t/libexternal-arm64e.dylib -objc_stubs_small
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs --syms \
# RUN:   --macho %t/small-arm64e.out | FileCheck %s --check-prefix=SMALL-ARM64E
# RUN: %lld -arch arm64e -lSystem -interposable -o %t/interposable-arm64e.out \
# RUN:   %t/main-arm64e.o %t/libexternal-arm64e.dylib -objc_stubs_fast
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs \
# RUN:   --macho %t/interposable-arm64e.out \
# RUN:   | FileCheck %s --check-prefixes=INTERPOSABLE,INTERPOSABLE-ARM64E

# A raw __got slot for _objc_msgSend must not change the slot the stubs use.
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-darwin %t/got-load.s \
# RUN:   -o %t/got-load-arm64e.o
# RUN: %lld -arch arm64e -lSystem -o %t/got-load-arm64e.out %t/got-load-arm64e.o \
# RUN:   %t/libexternal-arm64e.dylib -objc_stubs_fast -U _objc_msgSend
# RUN: llvm-objdump --no-show-raw-insn --section=__objc_stubs \
# RUN:   --section=__auth_got --section-headers --macho %t/got-load-arm64e.out \
# RUN:   | FileCheck %s --check-prefix=GOT-LOAD-ARM64E
# RUN: llvm-objdump --macho --dyld-info %t/got-load-arm64e.out \
# RUN:   | FileCheck %s --check-prefix=GOT-LOAD-ARM64E-GOT

# Small class stubs branch to _objc_msgSend's authenticated stub.
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-darwin %t/dynamic.s \
# RUN:   -o %t/dynamic-arm64e.o
# RUN: %lld -arch arm64e -lSystem -o %t/dynamic-small-arm64e.out \
# RUN:   %t/dynamic-arm64e.o -objc_stubs_small \
# RUN:   -U '_OBJC_CLASS_$_DynamicClass' -U _objc_msgSend
# RUN: llvm-objdump --no-show-raw-insn --section=__objc_stubs \
# RUN:   --section=__auth_stubs --section-headers --macho \
# RUN:   %t/dynamic-small-arm64e.out | FileCheck %s --check-prefix=DYNAMIC-SMALL-ARM64E

# Check archive extraction through class-stub references.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/archive-main.s \
# RUN:   -o %t/archive-main.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/archive-class.s \
# RUN:   -o %t/archive-class.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/archive-nested.s \
# RUN:   -o %t/archive-nested.o
# RUN: llvm-ar rcs %t/libarchive.a %t/archive-class.o %t/archive-nested.o
# RUN: %lld -arch arm64 -lSystem -o %t/archive.out %t/archive-main.o \
# RUN:   %t/libarchive.a -objc_stubs_fast -U _objc_msgSend
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs \
# RUN:   --macho %t/archive.out | FileCheck %s --check-prefix=ARCHIVE
# RUN: llvm-nm %t/archive.out | FileCheck %s --check-prefix=ARCHIVE-SYMS
# RUN: %lld -arch arm64 -lSystem -o %t/start-lib.out %t/archive-main.o \
# RUN:   --start-lib %t/archive-class.o %t/archive-nested.o --end-lib \
# RUN:   -objc_stubs_fast -U _objc_msgSend
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs \
# RUN:   --macho %t/start-lib.out | FileCheck %s --check-prefix=ARCHIVE
# RUN: llvm-nm %t/start-lib.out | FileCheck %s --check-prefix=ARCHIVE-SYMS

# Check autolink extraction when the archive member defines the class.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/autolink-main-a.s \
# RUN:   -o %t/autolink-main-a.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/autolink-dep-a.s \
# RUN:   -o %t/autolink-dep-a.o
# RUN: llvm-ar rcs %t/libautolinka.a %t/autolink-dep-a.o
# RUN: %lld -arch arm64 -lSystem -o %t/autolink-a.out \
# RUN:   %t/autolink-main-a.o -L%t -objc_stubs_fast -U _objc_msgSend
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs \
# RUN:   --macho %t/autolink-a.out | FileCheck %s --check-prefix=AUTOLINK-A

# Check autolink extraction when the class is already known.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/autolink-main-b.s \
# RUN:   -o %t/autolink-main-b.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/autolink-dep-b.s \
# RUN:   -o %t/autolink-dep-b.o
# RUN: llvm-ar rcs %t/libautolinkb.a %t/autolink-dep-b.o
# RUN: %lld -arch arm64 -lSystem -dead_strip -o %t/autolink-b.out \
# RUN:   %t/autolink-main-b.o -L%t -objc_stubs_fast -U _objc_msgSend
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs \
# RUN:   --macho %t/autolink-b.out | FileCheck %s --check-prefix=AUTOLINK-B
# RUN: llvm-nm %t/autolink-b.out | FileCheck %s --check-prefix=AUTOLINK-B-SYMS

# Check class stubs for dynamic class symbols.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/dynamic.s \
# RUN:   -o %t/dynamic.o
# RUN: %lld -arch arm64 -lSystem -o %t/dynamic.out %t/dynamic.o \
# RUN:   -objc_stubs_fast -U '_OBJC_CLASS_$_DynamicClass' -U _objc_msgSend
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs \
# RUN:   --macho %t/dynamic.out | FileCheck %s --check-prefix=DYNAMIC
# RUN: %lld -arch arm64 -lSystem -o %t/dynamic-lookup.out %t/dynamic.o \
# RUN:   -objc_stubs_fast -undefined dynamic_lookup
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs \
# RUN:   --macho %t/dynamic-lookup.out | FileCheck %s --check-prefix=DYNAMIC

# Check that a locally defined class that needs binding is loaded through the GOT.
# RUN: %lld -arch arm64 -lSystem -interposable -o %t/interposable.out \
# RUN:   %t/main.o %t/libexternal.dylib -objc_stubs_fast
# RUN: llvm-objdump --no-show-raw-insn --section=__TEXT,__objc_stubs \
# RUN:   --macho %t/interposable.out | FileCheck %s --check-prefix=INTERPOSABLE

# Check diagnostics for missing class symbols.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/missing.s \
# RUN:   -o %t/missing.o
# RUN: not %lld -arch arm64 -lSystem -o /dev/null %t/missing.o \
# RUN:   -objc_stubs_fast 2>&1 | FileCheck %s --check-prefix=MISSING

# Check diagnostics for malformed class-stub symbol names.
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/malformed.s \
# RUN:   -o %t/malformed.o
# RUN: not %lld -arch arm64 -lSystem -o /dev/null %t/malformed.o \
# RUN:   -objc_stubs_fast 2>&1 | FileCheck %s --check-prefix=MALFORMED

# FAST:      Contents of (__TEXT,__objc_stubs) section
# FAST:      _objc_msgSend$instance:
# FAST-NEXT: adrp    x1,
# FAST-NEXT: ldr     x1, {{.*}} ; Objc selector ref: instance
# FAST-NEXT: adrp    x16,
# FAST-NEXT: ldr     x16,
# FAST-NEXT: br      x16
# FAST-NEXT: brk     #0x1
# FAST-NEXT: brk     #0x1
# FAST-NEXT: brk     #0x1
# FAST-NEXT: _objc_msgSendClass$external$_OBJC_CLASS_$_ExternalClass:
# FAST-NEXT: adrp    x0,
# FAST-NEXT: ldr     x0, {{.*}} ; literal pool symbol address: _OBJC_CLASS_$_ExternalClass
# FAST-NEXT: adrp    x1,
# FAST-NEXT: ldr     x1, {{.*}} ; Objc selector ref: external
# FAST-NEXT: adrp    x16,
# FAST-NEXT: ldr     x16,
# FAST-NEXT: br      x16
# FAST-NEXT: brk     #0x1
# FAST-NEXT: _objc_msgSendClass$local$_OBJC_CLASS_$_LocalClass:
# FAST-NEXT: adrp    x0, [[#]] ; 0x[[#%x,FAST_LOCAL_CLASS_PAGE:]]
# FAST-NEXT: add     x0, x0, #0x[[#%x,FAST_LOCAL_CLASS_OFF:]]
# FAST-NEXT: adrp    x1,
# FAST-NEXT: ldr     x1, {{.*}} ; Objc selector ref: local
# FAST-NEXT: adrp    x16,
# FAST-NEXT: ldr     x16,
# FAST-NEXT: br      x16
# FAST-NEXT: brk     #0x1
# FAST-LABEL: SYMBOL TABLE:
# FAST-DAG: {{0*}}[[#%x,FAST_LOCAL_CLASS_PAGE+FAST_LOCAL_CLASS_OFF]] g     O __DATA,__data _OBJC_CLASS_$_LocalClass

# SMALL:      Contents of (__TEXT,__objc_stubs) section
# SMALL:      _objc_msgSend$instance:
# SMALL-NEXT: adrp    x1,
# SMALL-NEXT: ldr     x1, {{.*}} ; Objc selector ref: instance
# SMALL-NEXT: b       _objc_msgSend
# SMALL-NEXT: _objc_msgSendClass$external$_OBJC_CLASS_$_ExternalClass:
# SMALL-NEXT: adrp    x0,
# SMALL-NEXT: ldr     x0, {{.*}} ; literal pool symbol address: _OBJC_CLASS_$_ExternalClass
# SMALL-NEXT: adrp    x1,
# SMALL-NEXT: ldr     x1, {{.*}} ; Objc selector ref: external
# SMALL-NEXT: b       _objc_msgSend
# SMALL-NEXT: _objc_msgSendClass$local$_OBJC_CLASS_$_LocalClass:
# SMALL-NEXT: adrp    x0, [[#]] ; 0x[[#%x,SMALL_LOCAL_CLASS_PAGE:]]
# SMALL-NEXT: add     x0, x0, #0x[[#%x,SMALL_LOCAL_CLASS_OFF:]]
# SMALL-NEXT: adrp    x1,
# SMALL-NEXT: ldr     x1, {{.*}} ; Objc selector ref: local
# SMALL-NEXT: b       _objc_msgSend
# SMALL-LABEL: SYMBOL TABLE:
# SMALL-DAG: {{0*}}[[#%x,SMALL_LOCAL_CLASS_PAGE+SMALL_LOCAL_CLASS_OFF]] g     O __DATA,__data _OBJC_CLASS_$_LocalClass

# Each arm64e tail is pinned to the address of __auth_got: __auth_got and __got
# can share a page, so a stub that read the raw __got would otherwise match.
# llvm-objdump does not decode chained fixups yet, so most selector refs are
# not annotated on arm64e and are left unchecked.
# FAST-ARM64E:      __auth_got {{[0-9a-f]+}} {{0*}}[[#%x,FAST_E_AUTH_GOT:]] DATA
# FAST-ARM64E:      Contents of (__TEXT,__objc_stubs) section
# FAST-ARM64E:      _objc_msgSend$instance:
# FAST-ARM64E-NEXT: adrp    x1,
# FAST-ARM64E-NEXT: ldr     x1, [x1
# FAST-ARM64E-NEXT: adrp    x17, [[#]] ; 0x[[#%x,FAST_E_INSTANCE_PAGE:]]
# FAST-ARM64E-NEXT: add     x17, x17, #0x[[#%x,FAST_E_AUTH_GOT-FAST_E_INSTANCE_PAGE]]
# FAST-ARM64E-NEXT: ldr     x16, [x17]
# FAST-ARM64E-NEXT: braa    x16, x17
# FAST-ARM64E-NEXT: brk     #0x1
# FAST-ARM64E-NEXT: brk     #0x1
# FAST-ARM64E-NEXT: _objc_msgSendClass$external$_OBJC_CLASS_$_ExternalClass:
# FAST-ARM64E-NEXT: adrp    x0,
# FAST-ARM64E-NEXT: ldr     x0, {{.*}} ; literal pool symbol address: _OBJC_CLASS_$_ExternalClass
# FAST-ARM64E-NEXT: adrp    x1,
# FAST-ARM64E-NEXT: ldr     x1, [x1
# FAST-ARM64E-NEXT: adrp    x17, [[#]] ; 0x[[#%x,FAST_E_EXTERNAL_PAGE:]]
# FAST-ARM64E-NEXT: add     x17, x17, #0x[[#%x,FAST_E_AUTH_GOT-FAST_E_EXTERNAL_PAGE]]
# FAST-ARM64E-NEXT: ldr     x16, [x17]
# FAST-ARM64E-NEXT: braa    x16, x17
# FAST-ARM64E-NEXT: _objc_msgSendClass$local$_OBJC_CLASS_$_LocalClass:
# FAST-ARM64E-NEXT: adrp    x0, [[#]] ; 0x[[#%x,FAST_E_LOCAL_CLASS_PAGE:]]
# FAST-ARM64E-NEXT: add     x0, x0, #0x[[#%x,FAST_E_LOCAL_CLASS_OFF:]]
# FAST-ARM64E-NEXT: adrp    x1,
# FAST-ARM64E-NEXT: ldr     x1, [x1
# FAST-ARM64E-NEXT: adrp    x17, [[#]] ; 0x[[#%x,FAST_E_LOCAL_PAGE:]]
# FAST-ARM64E-NEXT: add     x17, x17, #0x[[#%x,FAST_E_AUTH_GOT-FAST_E_LOCAL_PAGE]]
# FAST-ARM64E-NEXT: ldr     x16, [x17]
# FAST-ARM64E-NEXT: braa    x16, x17
# FAST-ARM64E-LABEL: SYMBOL TABLE:
# FAST-ARM64E-DAG: {{0*}}[[#%x,FAST_E_LOCAL_CLASS_PAGE+FAST_E_LOCAL_CLASS_OFF]] g     O __DATA,__data _OBJC_CLASS_$_LocalClass

# The class pointer stays unsigned in the raw __got, as in ld64.
# FAST-ARM64E-GOT: __DATA_CONST __auth_got 0x{{[0-9A-Fa-f]+}} 0x{{[0-9A-Fa-f]+}} rebase 0x0000 true IA
# FAST-ARM64E-GOT: __DATA_CONST __got 0x{{[0-9A-Fa-f]+}} 0x{{[0-9A-Fa-f]+}} bind 0x0 libexternal _OBJC_CLASS_$_ExternalClass

# SMALL-ARM64E:      Contents of (__TEXT,__objc_stubs) section
# SMALL-ARM64E:      _objc_msgSend$instance:
# SMALL-ARM64E-NEXT: adrp    x1,
# SMALL-ARM64E-NEXT: ldr     x1, [x1
# SMALL-ARM64E-NEXT: b       _objc_msgSend
# SMALL-ARM64E-NEXT: _objc_msgSendClass$external$_OBJC_CLASS_$_ExternalClass:
# SMALL-ARM64E-NEXT: adrp    x0,
# SMALL-ARM64E-NEXT: ldr     x0, {{.*}} ; literal pool symbol address: _OBJC_CLASS_$_ExternalClass
# SMALL-ARM64E-NEXT: adrp    x1,
# SMALL-ARM64E-NEXT: ldr     x1, [x1
# SMALL-ARM64E-NEXT: b       _objc_msgSend
# SMALL-ARM64E-NEXT: _objc_msgSendClass$local$_OBJC_CLASS_$_LocalClass:
# SMALL-ARM64E-NEXT: adrp    x0, [[#]] ; 0x[[#%x,SMALL_E_LOCAL_CLASS_PAGE:]]
# SMALL-ARM64E-NEXT: add     x0, x0, #0x[[#%x,SMALL_E_LOCAL_CLASS_OFF:]]
# SMALL-ARM64E-NEXT: adrp    x1,
# SMALL-ARM64E-NEXT: ldr     x1, [x1
# SMALL-ARM64E-NEXT: b       _objc_msgSend
# SMALL-ARM64E-LABEL: SYMBOL TABLE:
# SMALL-ARM64E-DAG: {{0*}}[[#%x,SMALL_E_LOCAL_CLASS_PAGE+SMALL_E_LOCAL_CLASS_OFF]] g     O __DATA,__data _OBJC_CLASS_$_LocalClass

# GOT-LOAD-ARM64E:      __auth_got {{[0-9a-f]+}} {{0*}}[[#%x,GOT_LOAD_AUTH_GOT:]] DATA
# GOT-LOAD-ARM64E:      Contents of (__TEXT,__objc_stubs) section
# GOT-LOAD-ARM64E-NEXT: _objc_msgSend$got:
# GOT-LOAD-ARM64E-NEXT: adrp    x1,
# GOT-LOAD-ARM64E-NEXT: ldr     x1, [x1
# GOT-LOAD-ARM64E-NEXT: adrp    x17, [[#]] ; 0x[[#%x,GOT_LOAD_SEL_PAGE:]]
# GOT-LOAD-ARM64E-NEXT: add     x17, x17, #0x[[#%x,GOT_LOAD_AUTH_GOT-GOT_LOAD_SEL_PAGE]]
# GOT-LOAD-ARM64E-NEXT: ldr     x16, [x17]
# GOT-LOAD-ARM64E-NEXT: braa    x16, x17
# GOT-LOAD-ARM64E-NEXT: brk     #0x1
# GOT-LOAD-ARM64E-NEXT: brk     #0x1
# GOT-LOAD-ARM64E-NEXT: _objc_msgSendClass$got$_OBJC_CLASS_$_ExternalClass:
# GOT-LOAD-ARM64E-NEXT: adrp    x0,
# GOT-LOAD-ARM64E-NEXT: ldr     x0, {{.*}} ; literal pool symbol address: _OBJC_CLASS_$_ExternalClass
# GOT-LOAD-ARM64E-NEXT: adrp    x1,
# GOT-LOAD-ARM64E-NEXT: ldr     x1, [x1
# GOT-LOAD-ARM64E-NEXT: adrp    x17, [[#]] ; 0x[[#%x,GOT_LOAD_CLASS_PAGE:]]
# GOT-LOAD-ARM64E-NEXT: add     x17, x17, #0x[[#%x,GOT_LOAD_AUTH_GOT-GOT_LOAD_CLASS_PAGE]]
# GOT-LOAD-ARM64E-NEXT: ldr     x16, [x17]
# GOT-LOAD-ARM64E-NEXT: braa    x16, x17

# _objc_msgSend has an authenticated bind in __auth_got for the stubs and a
# plain one in __got for the GOT load.
# GOT-LOAD-ARM64E-GOT: __DATA_CONST __auth_got 0x{{[0-9A-Fa-f]+}} 0x{{[0-9A-Fa-f]+}} bind 0x0000 true IA 0x0 flat-namespace _objc_msgSend
# GOT-LOAD-ARM64E-GOT: __DATA_CONST __got 0x{{[0-9A-Fa-f]+}} 0x{{[0-9A-Fa-f]+}} bind 0x0 flat-namespace _objc_msgSend

# DYNAMIC-SMALL-ARM64E:      __auth_stubs {{[0-9a-f]+}} {{0*}}[[#%x,DYN_E_AUTH_STUBS:]] TEXT
# DYNAMIC-SMALL-ARM64E:      _objc_msgSendClass$dynamic$_OBJC_CLASS_$_DynamicClass:
# DYNAMIC-SMALL-ARM64E-NEXT: adrp    x0,
# DYNAMIC-SMALL-ARM64E-NEXT: ldr     x0, {{.*}} ; literal pool symbol address: _OBJC_CLASS_$_DynamicClass
# DYNAMIC-SMALL-ARM64E-NEXT: adrp    x1,
# DYNAMIC-SMALL-ARM64E-NEXT: ldr     x1, [x1
# DYNAMIC-SMALL-ARM64E-NEXT: b       0x[[#%x,DYN_E_AUTH_STUBS]]

# DYNAMIC:      Contents of (__TEXT,__objc_stubs) section
# DYNAMIC-NEXT: _objc_msgSendClass$dynamic$_OBJC_CLASS_$_DynamicClass:
# DYNAMIC-NEXT: adrp    x0,
# DYNAMIC-NEXT: ldr     x0, {{.*}} ; literal pool symbol address: _OBJC_CLASS_$_DynamicClass
# DYNAMIC-NEXT: adrp    x1,
# DYNAMIC-NEXT: ldr     x1, {{.*}} ; Objc selector ref: dynamic
# DYNAMIC-NEXT: adrp    x16,
# DYNAMIC-NEXT: ldr     x16, {{.*}} ; literal pool symbol address: _objc_msgSend
# DYNAMIC-NEXT: br      x16
# DYNAMIC-NEXT: brk     #0x1

# INTERPOSABLE:      _objc_msgSendClass$local$_OBJC_CLASS_$_LocalClass:
# INTERPOSABLE-NEXT: adrp    x0,
# INTERPOSABLE-NEXT: ldr     x0, {{.*}} ; literal pool symbol address: _OBJC_CLASS_$_LocalClass
# _objc_msgSend is not the first __auth_got entry here and has no __got slot,
# so the annotation shows the stub uses its own __auth_got slot.
# INTERPOSABLE-ARM64E-NEXT: adrp    x1,
# INTERPOSABLE-ARM64E-NEXT: ldr     x1, [x1
# INTERPOSABLE-ARM64E-NEXT: adrp    x17,
# INTERPOSABLE-ARM64E-NEXT: add     x17, x17, #0x{{[0-9a-f]+}} ; literal pool symbol address: _objc_msgSend{{$}}
# INTERPOSABLE-ARM64E-NEXT: ldr     x16, [x17]
# INTERPOSABLE-ARM64E-NEXT: braa    x16, x17

# ARCHIVE:      Contents of (__TEXT,__objc_stubs) section
# ARCHIVE-NEXT: _objc_msgSendClass$archive$_OBJC_CLASS_$_ArchiveClass:
# ARCHIVE-NEXT: adrp    x0,
# ARCHIVE-NEXT: add     x0, x0,
# ARCHIVE-NEXT: adrp    x1,
# ARCHIVE-NEXT: ldr     x1, {{.*}} ; Objc selector ref: archive
# ARCHIVE-NEXT: adrp    x16,
# ARCHIVE-NEXT: ldr     x16, {{.*}} ; literal pool symbol address: _objc_msgSend
# ARCHIVE-NEXT: br      x16
# ARCHIVE-NEXT: brk     #0x1
# ARCHIVE-NEXT: _objc_msgSendClass$nested$_OBJC_CLASS_$_NestedClass:
# ARCHIVE-NEXT: adrp    x0,
# ARCHIVE-NEXT: add     x0, x0,
#
# ARCHIVE-SYMS: _OBJC_CLASS_$_ArchiveClass
# ARCHIVE-SYMS: _OBJC_CLASS_$_NestedClass

# AUTOLINK-A:      Contents of (__TEXT,__objc_stubs) section
# AUTOLINK-A-NEXT: _objc_msgSendClass$auto$_OBJC_CLASS_$_AutoClass:
# AUTOLINK-A-NEXT: adrp    x0,
# AUTOLINK-A-NEXT: add     x0, x0,
# AUTOLINK-A-NEXT: adrp    x1,
# AUTOLINK-A-NEXT: ldr     x1, {{.*}} ; Objc selector ref: auto
# AUTOLINK-A-NEXT: adrp    x16,
# AUTOLINK-A-NEXT: ldr     x16, {{.*}} ; literal pool symbol address: _objc_msgSend
# AUTOLINK-A-NEXT: br      x16
# AUTOLINK-A-NEXT: brk     #0x1
#
# AUTOLINK-B:      Contents of (__TEXT,__objc_stubs) section
# AUTOLINK-B-NEXT: _objc_msgSendClass$late$_OBJC_CLASS_$_LateClass:
# AUTOLINK-B-NEXT: adrp    x0,
# AUTOLINK-B-NEXT: add     x0, x0,
# AUTOLINK-B-NEXT: adrp    x1,
# AUTOLINK-B-NEXT: ldr     x1, {{.*}} ; Objc selector ref: late
# AUTOLINK-B-NEXT: adrp    x16,
# AUTOLINK-B-NEXT: ldr     x16, {{.*}} ; literal pool symbol address: _objc_msgSend
# AUTOLINK-B-NEXT: br      x16
# AUTOLINK-B-NEXT: brk     #0x1
#
# AUTOLINK-B-SYMS: _OBJC_CLASS_$_LateClass

# MISSING: error: undefined symbol: _OBJC_CLASS_$_MissingClass
# MISSING-NEXT: >>> referenced by objc class stub

# MALFORMED: error: malformed objc class stub symbol _objc_msgSendClass$malformed; expected _objc_msgSendClass$<selector>$_OBJC_CLASS_$_<class>
# MALFORMED-NOT: Objc selector ref:
# MALFORMED-NOT: undefined symbol: _objc_msgSendClass$malformed

#--- main.s
# Main fast/small case: one regular selector stub plus local and dylib class
# stubs.
.section __TEXT,__objc_methname,cstring_literals
Linstance:
  .asciz "instance"
Llocal:
  .asciz "local"
Lexternal:
  .asciz "external"

.section __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align 3
  .quad Linstance
  .quad Llocal
  .quad Lexternal

.text
.globl _objc_msgSend
_objc_msgSend:
  ret

.globl _main
_main:
  bl _objc_msgSend$instance
  bl _objc_msgSendClass$local$_OBJC_CLASS_$_LocalClass
  bl _objc_msgSendClass$external$_OBJC_CLASS_$_ExternalClass
  ret

.data
.globl _OBJC_CLASS_$_LocalClass
_OBJC_CLASS_$_LocalClass:
  .quad 0

.subsections_via_symbols

#--- external.s
# Provides the dylib-defined class used by main.s.
.data
.globl _OBJC_CLASS_$_ExternalClass
_OBJC_CLASS_$_ExternalClass:
  .quad 0

#--- dynamic.s
# Uses a class stub whose class symbol is resolved dynamically.
.section __TEXT,__objc_methname,cstring_literals
Ldynamic:
  .asciz "dynamic"

.section __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align 3
  .quad Ldynamic

.text
.globl _main
_main:
  bl _objc_msgSendClass$dynamic$_OBJC_CLASS_$_DynamicClass
  ret

#--- archive-main.s
# References a class stub whose class definition must be pulled from an archive.
.text
.globl _main
_main:
  bl _objc_msgSendClass$archive$_OBJC_CLASS_$_ArchiveClass
  ret

#--- archive-class.s
# Archive member defining the class used by archive-main.s. Its class stub
# extracts another member.
.text
.globl _archive_impl
_archive_impl:
  bl _objc_msgSendClass$nested$_OBJC_CLASS_$_NestedClass
  ret

.data
.globl _OBJC_CLASS_$_ArchiveClass
_OBJC_CLASS_$_ArchiveClass:
  .quad 0

#--- archive-nested.s
# Archive member defining the class used by archive-class.s.
.data
.globl _OBJC_CLASS_$_NestedClass
_OBJC_CLASS_$_NestedClass:
  .quad 0

#--- autolink-main-a.s
# Autolinks an archive member that contains both the stub user and class.
.linker_option "-lautolinka"
.text
.globl _main
_main:
  bl _dep_a
  ret

#--- autolink-dep-a.s
# Autolinked member containing the class-stub reference and class definition.
.text
.globl _dep_a
_dep_a:
  bl _objc_msgSendClass$auto$_OBJC_CLASS_$_AutoClass
  ret

.data
.globl _OBJC_CLASS_$_AutoClass
_OBJC_CLASS_$_AutoClass:
  .quad 0

#--- autolink-main-b.s
# Autolinks an archive member that references a class defined in this object.
.linker_option "-lautolinkb"
.text
.globl _main
_main:
  bl _dep_b
  ret

.data
.globl _OBJC_CLASS_$_LateClass
_OBJC_CLASS_$_LateClass:
  .quad 0

#--- autolink-dep-b.s
# Autolinked member that introduces a late class-stub reference.
.text
.globl _dep_b
_dep_b:
  bl _objc_msgSendClass$late$_OBJC_CLASS_$_LateClass
  ret

#--- missing.s
# References a live class stub whose class symbol is unresolved.
.section __TEXT,__objc_methname,cstring_literals
Lmissing:
  .asciz "missing"

.section __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align 3
  .quad Lmissing

.text
.globl _objc_msgSend
_objc_msgSend:
  ret

.globl _main
_main:
  bl _objc_msgSendClass$missing$_OBJC_CLASS_$_MissingClass
  ret

#--- malformed.s
# References a class stub symbol without the required class-symbol suffix.
.text
.globl _main
_main:
  bl _objc_msgSendClass$malformed
  ret

#--- got-load.s
# Takes the address of _objc_msgSend through a GOT load, as clang does on
# arm64e, which gives it a raw __got slot next to its __auth_got slot.
.section __TEXT,__objc_methname,cstring_literals
Lgot:
  .asciz "got"

.section __DATA,__objc_selrefs,literal_pointers,no_dead_strip
.p2align 3
  .quad Lgot

.text
.globl _main
_main:
  adrp x8, _objc_msgSend@GOTPAGE
  ldr x8, [x8, _objc_msgSend@GOTPAGEOFF]
  bl _objc_msgSend$got
  bl _objc_msgSendClass$got$_OBJC_CLASS_$_ExternalClass
  ret

.subsections_via_symbols
