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
# UNKNOWN: error: {{.*}}v2.o has architecture unknown which is incompatible with target architecture arm64e

## Without -arch_errors_fatal, the object is ignored with a warning.
# RUN: not %no-arg-lld -arch arm64e -platform_version ios 17.0 17.0 %t/main.o %t/v1.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN
# WARN: warning: {{.*}}v1.o has architecture arm64e.v1 which is incompatible with target architecture arm64e
# WARN: error: undefined symbol: _helper

## ld64 accepts some other markers in static links, such as kernels.
# RUN: %no-arg-lld -arch arm64e -platform_version ios 17.0 17.0 -static -e _main %t/main.o %t/kernel.o -o %t/static

#--- main.s
.globl _main
.p2align 2
_main:
  bl _helper
  ret

#--- helper.yaml
## An object defining _helper; SUBTYPE varies only its arm64e ABI marker.
--- !mach-o
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
