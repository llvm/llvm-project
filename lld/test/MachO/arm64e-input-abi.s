# REQUIRES: aarch64
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-darwin %t/main.s -o %t/main.o
# RUN: yaml2obj -DSUBTYPE=0x80000002 %t/helper.yaml -o %t/helper.o
# RUN: yaml2obj -DSUBTYPE=0x2 %t/helper.yaml -o %t/old.o
# RUN: yaml2obj -DSUBTYPE=0x81000002 %t/helper.yaml -o %t/v1.o
# RUN: yaml2obj -DSUBTYPE=0xC0000002 %t/helper.yaml -o %t/kernel.o
# RUN: yaml2obj -DSUBTYPE=0xC1000002 %t/helper.yaml -o %t/kernel-v1.o
# RUN: yaml2obj -DSUBTYPE=0x82000002 %t/helper.yaml -o %t/v2.o
# RUN: llvm-ar rcs %t/old.a %t/old.o
# RUN: llvm-ar rcs %t/v1.a %t/v1.o
# RUN: llvm-ar rcs %t/kernel.a %t/kernel.o

## Objects must be marked userland ABI version 0, even on platforms whose
## executables are marked version 1.
# RUN: %no-arg-lld -arch arm64e -platform_version ios 17.0 17.0 %t/main.o %t/helper.o -o %t/ios
# RUN: %no-arg-lld -arch arm64e -platform_version macos 14.0 14.0 %t/main.o %t/helper.o -o %t/mac
# RUN: %no-arg-lld -arch arm64e -platform_version mac-catalyst 17.0 17.0 %t/main.o %t/helper.o -o %t/catalyst

## Like ld64, treat any other marker as another architecture, in both thin
## objects and archive members.
# RUN: not %lld -arch arm64e -arch_errors_fatal %t/main.o %t/old.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=OLD
# RUN: not %lld -arch arm64e -arch_errors_fatal %t/main.o %t/old.a -o /dev/null 2>&1 | FileCheck %s --check-prefix=OLD
# RUN: not %lld -arch arm64e -arch_errors_fatal %t/main.o %t/v1.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=V1
# RUN: not %lld -arch arm64e -arch_errors_fatal %t/main.o %t/v1.a -o /dev/null 2>&1 | FileCheck %s --check-prefix=V1
# RUN: not %lld -arch arm64e -arch_errors_fatal %t/main.o %t/kernel.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=KERNEL
# RUN: not %lld -arch arm64e -arch_errors_fatal %t/main.o %t/kernel.a -o /dev/null 2>&1 | FileCheck %s --check-prefix=KERNEL
# RUN: not %lld -arch arm64e -arch_errors_fatal %t/main.o %t/kernel-v1.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=KERNEL-V1
# RUN: not %lld -arch arm64e -arch_errors_fatal %t/main.o %t/v2.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNKNOWN
# RUN: not %no-arg-lld -arch arm64e -arch_errors_fatal -platform_version mac-catalyst 17.0 17.0 %t/main.o %t/v1.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=V1

# OLD: error: {{.*}}old.{{.*}} has architecture arm64e.old which is incompatible with target architecture arm64e
# V1: error: {{.*}}v1.{{.*}} has architecture arm64e.v1 which is incompatible with target architecture arm64e
# KERNEL: error: {{.*}}kernel.{{.*}} has architecture arm64e.kernel which is incompatible with target architecture arm64e
# KERNEL-V1: error: {{.*}}kernel-v1.o has architecture arm64e.kernel.v1 which is incompatible with target architecture arm64e
# UNKNOWN: error: {{.*}}v2.{{.*}} has architecture unknown which is incompatible with target architecture arm64e

## Without -arch_errors_fatal, the object is ignored with a warning.
# RUN: not %no-arg-lld -arch arm64e -platform_version ios 17.0 17.0 %t/main.o %t/v1.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN
# WARN: warning: {{.*}}v1.o has architecture arm64e.v1 which is incompatible with target architecture arm64e
# WARN: error: undefined symbol: _helper

## Static links, like ld64's, also take unversioned and kernel objects, but no
## other markers.
# RUN: %no-arg-lld -arch arm64e -platform_version ios 17.0 17.0 -static -e _main %t/main.o %t/kernel.o -o %t/static
# RUN: %no-arg-lld -arch arm64e -platform_version ios 17.0 17.0 -static -e _main %t/main.o %t/old.o -o %t/static
# RUN: not %no-arg-lld -arch arm64e -arch_errors_fatal -platform_version ios 17.0 17.0 -static -e _main %t/main.o %t/v1.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=V1
# RUN: not %no-arg-lld -arch arm64e -arch_errors_fatal -platform_version ios 17.0 17.0 -static -e _main %t/main.o %t/kernel-v1.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=KERNEL-V1
# RUN: not %no-arg-lld -arch arm64e -arch_errors_fatal -platform_version ios 17.0 17.0 -static -e _main %t/main.o %t/v2.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNKNOWN

## Dylibs and bundle loaders are checked too, but more loosely. Like ld64, an
## arm64e link also takes those marked version 1, as macOS executables are, and
## arm64e.x1 dylibs marked version 0. An arm64 link takes any arm64e dylib, and
## an arm64e link any arm64-only TBD.
# RUN: yaml2obj -DSUBTYPE=0x80000002 %t/libhelper.yaml -o %t/libv0.dylib
# RUN: yaml2obj -DSUBTYPE=0x81000002 %t/libhelper.yaml -o %t/libv1.dylib
# RUN: yaml2obj -DSUBTYPE=0x8000000C %t/libhelper.yaml -o %t/libx1.dylib
# RUN: yaml2obj -DSUBTYPE=0x0 %t/libhelper.yaml -o %t/libarm64.dylib
# RUN: yaml2obj -DSUBTYPE=0x2 %t/libhelper.yaml -o %t/libold.dylib
# RUN: yaml2obj -DSUBTYPE=0xC0000002 %t/libhelper.yaml -o %t/libkernel.dylib
# RUN: yaml2obj -DSUBTYPE=0x82000002 %t/libhelper.yaml -o %t/libv2.dylib
# RUN: llvm-lipo -create %t/libarm64.dylib %t/libkernel.dylib -output %t/libfat-kernel.dylib
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/main.s -o %t/main-arm64.o
# RUN: %lld -arch arm64e %t/main.o %t/libv0.dylib -o %t/dylib
# RUN: %lld -arch arm64e %t/main.o %t/libv1.dylib -o %t/dylib
# RUN: %lld -arch arm64e %t/main.o %t/libx1.dylib -o %t/dylib
# RUN: %lld -arch arm64 %t/main-arm64.o %t/libkernel.dylib -o %t/dylib
# RUN: %lld -arch arm64e %t/main.o %t/libhelper.tbd -o %t/dylib
# RUN: %no-arg-lld -arch arm64e -platform_version macos 14.0 14.0 -e _helper %t/helper.o -o %t/loader
# RUN: %no-arg-lld -arch arm64e -platform_version macos 14.0 14.0 -bundle -bundle_loader %t/loader %t/main.o -o %t/bundle

## Other dylibs are ignored as another architecture, including the arm64e slice
## of a fat dylib, and get no load command.
# RUN: not %lld -arch arm64e -arch_errors_fatal %t/main.o %t/libarm64.dylib -o /dev/null 2>&1 | FileCheck %s --check-prefix=DYLIB-ARM64
# RUN: not %lld -arch arm64e -arch_errors_fatal %t/main.o %t/libold.dylib -o /dev/null 2>&1 | FileCheck %s --check-prefix=OLD
# RUN: not %lld -arch arm64e -arch_errors_fatal %t/main.o %t/libkernel.dylib -o /dev/null 2>&1 | FileCheck %s --check-prefix=KERNEL
# RUN: not %lld -arch arm64e -arch_errors_fatal %t/main.o %t/libv2.dylib -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNKNOWN
# RUN: not %lld -arch arm64e -arch_errors_fatal %t/main.o %t/libfat-kernel.dylib -o /dev/null 2>&1 | FileCheck %s --check-prefix=KERNEL
# RUN: %no-fatal-warnings-lld -arch arm64e %t/main.o %t/helper.o %t/libarm64.dylib -o %t/ignored 2>&1 | FileCheck %s --check-prefix=WARN-DYLIB
# RUN: llvm-otool -L %t/ignored | FileCheck %s --check-prefix=NO-LOAD

# DYLIB-ARM64: error: {{.*}}libarm64.dylib has architecture arm64 which is incompatible with target architecture arm64e
# WARN-DYLIB: warning: {{.*}}libarm64.dylib has architecture arm64 which is incompatible with target architecture arm64e
# NO-LOAD: ignored:
# NO-LOAD-NOT: libhelper.dylib

## Bitcode is checked by its target triple, in either input order. LTO would
## otherwise compile arm64 bitcode into an arm64e image without pointer
## authentication, or fail.
# RUN: llvm-as %t/main.ll -o %t/main.bc
# RUN: llvm-as %t/helper.ll -o %t/helper.bc
# RUN: opt -mtriple=arm64-apple-darwin %t/helper.ll -o %t/helper-arm64.bc
# RUN: llvm-ar rcs %t/helper-arm64.a %t/helper-arm64.bc
# RUN: %lld -arch arm64e %t/main.bc %t/helper.bc -o %t/lto
# RUN: not %lld -arch arm64e -arch_errors_fatal %t/main.bc %t/helper-arm64.bc -o /dev/null 2>&1 | FileCheck %s --check-prefix=BITCODE
# RUN: not %lld -arch arm64e -arch_errors_fatal %t/helper-arm64.bc %t/main.bc -o /dev/null 2>&1 | FileCheck %s --check-prefix=BITCODE
# RUN: not %lld -arch arm64e -arch_errors_fatal %t/main.bc %t/helper-arm64.a -o /dev/null 2>&1 | FileCheck %s --check-prefix=BITCODE
# RUN: not %lld -arch arm64 -arch_errors_fatal %t/main-arm64.o %t/helper.bc -o /dev/null 2>&1 | FileCheck %s --check-prefix=BITCODE-ARM64E
# RUN: not %no-fatal-warnings-lld -arch arm64e %t/main.bc %t/helper-arm64.bc -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN-BITCODE

# BITCODE: error: {{.*}}helper-arm64.{{.*}} has architecture arm64 which is incompatible with target architecture arm64e
# BITCODE-ARM64E: error: {{.*}}helper.bc has architecture arm64e which is incompatible with target architecture arm64
# WARN-BITCODE: warning: {{.*}}helper-arm64.bc has architecture arm64 which is incompatible with target architecture arm64e
# WARN-BITCODE: error: undefined symbol: _helper

## -ObjC does not load an ignored lazy file with an ObjC category either.
# RUN: opt -mtriple=arm64-apple-darwin %t/objc.ll -o %t/objc-arm64.bc
# RUN: not %no-fatal-warnings-lld -arch arm64e -ObjC %t/main.o --start-lib %t/objc-arm64.bc --end-lib -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN-OBJC
# WARN-OBJC: warning: {{.*}}objc-arm64.bc has architecture arm64 which is incompatible with target architecture arm64e
# WARN-OBJC: error: undefined symbol: _helper

#--- main.s
.globl _main
.p2align 2
_main:
  bl _helper
  ret

#--- helper.yaml
## An object defining _helper; SUBTYPE varies only its arm64e ABI marker.
--- !mach-o
IsLittleEndian:  true
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x100000C
  cpusubtype:      [[SUBTYPE]]
  filetype:        0x1
  ncmds:           3
  sizeofcmds:      256
  flags:           0x0
  reserved:        0x0
LoadCommands:
  - cmd:             LC_SEGMENT_64
    cmdsize:         152
    segname:         ''
    vmaddr:          0
    vmsize:          4
    fileoff:         288
    filesize:        4
    maxprot:         7
    initprot:        7
    nsects:          1
    flags:           0
    Sections:
      - sectname:        __text
        segname:         __TEXT
        addr:            0x0
        size:            4
        offset:          0x120
        align:           2
        reloff:          0x0
        nreloc:          0
        flags:           0x80000400
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         C0035FD6
  - cmd:             LC_SYMTAB
    cmdsize:         24
    symoff:          296
    nsyms:           2
    stroff:          328
    strsize:         16
  - cmd:             LC_DYSYMTAB
    cmdsize:         80
    ilocalsym:       0
    nlocalsym:       1
    iextdefsym:      1
    nextdefsym:      1
    iundefsym:       2
    nundefsym:       0
    tocoff:          0
    ntoc:            0
    modtaboff:       0
    nmodtab:         0
    extrefsymoff:    0
    nextrefsyms:     0
    indirectsymoff:  0
    nindirectsyms:   0
    extreloff:       0
    nextrel:         0
    locreloff:       0
    nlocrel:         0
LinkEditData:
  NameList:
    - n_strx:          9
      n_type:          0xE
      n_sect:          1
      n_desc:          0
      n_value:         0
    - n_strx:          1
      n_type:          0xF
      n_sect:          1
      n_desc:          0
      n_value:         0
  StringTable:
    - ''
    - _helper
    - ltmp0
    - ''
...

#--- libhelper.yaml
## A dylib exporting _helper; SUBTYPE varies its CPU subtype.
--- !mach-o
IsLittleEndian:  true
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x100000C
  cpusubtype:      [[SUBTYPE]]
  filetype:        0x6
  ncmds:           2
  sizeofcmds:      64
  flags:           0x0
  reserved:        0x0
LoadCommands:
  - cmd:             LC_ID_DYLIB
    cmdsize:         48
    dylib:
      name:            24
      timestamp:       0
      current_version: 0
      compatibility_version: 0
    Content:         '@rpath/libhelper.dylib'
    ZeroPadBytes:    2
  - cmd:             LC_DYLD_EXPORTS_TRIE
    cmdsize:         16
    dataoff:         96
    datasize:        17
LinkEditData:
  ExportTrie:
    TerminalSize:    0
    NodeOffset:      0
    Name:            ''
    Flags:           0x0
    Address:         0x0
    Other:           0x0
    ImportName:      ''
    Children:
      - TerminalSize:    4
        NodeOffset:      11
        Name:            _helper
        Flags:           0x0
        Address:         0x4000
        Other:           0x0
        ImportName:      ''
...

#--- libhelper.tbd
--- !tapi-tbd
tbd-version:     4
targets:         [ arm64-macos ]
install-name:    '@rpath/libhelper.dylib'
exports:
  - targets:         [ arm64-macos ]
    symbols:         [ _helper ]
...

#--- main.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64e-apple-darwin"

declare void @helper()

define void @main() {
  call void @helper()
  ret void
}

#--- helper.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64e-apple-darwin"

define void @helper() {
  ret void
}

#--- objc.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64e-apple-darwin"

@category = private global i8 0
@catlist = private global [1 x ptr] [ptr @category], section "__DATA,__objc_catlist,regular,no_dead_strip"

define void @helper() {
  ret void
}
