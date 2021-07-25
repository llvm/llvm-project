# REQUIRES: x86

## Test that we correctly handle symbol relocations in the compact unwind
## section.

## llvm-mc does not emit such relocations for compact unwind, but `ld -r` does.
## As such, these yaml files were from an object file produced with 'ld -r',
## specifically:
##
##  // foo.s
## .text
## .globl _main
## _main:
##   .cfi_startproc
##   .cfi_def_cfa_offset 16
##   .cfi_endproc
##   nop
##
## llvm-mc -filetype=obj -triple=x86_64-apple-macos10.15 -o foo1.o foo.s
## ld -r -o foo.o foo1.o

# RUN: rm -rf %t; mkdir -p %t
# RUN: yaml2obj %s -o %t/foo.o
# RUN: %lld -o %t/a.out %t/foo.o
# RUN: llvm-objdump --macho --section-headers %t/a.out | FileCheck %s
# CHECK: __unwind_info {{.*}} DATA

--- !mach-o
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x01000007
  cpusubtype:      0x00000003
  filetype:        0x00000001
  ncmds:           2
  sizeofcmds:      384
  flags:           0x00000000
  reserved:        0x00000000
LoadCommands:
  - cmd:             LC_SEGMENT_64
    cmdsize:         312
    segname:         ''
    vmaddr:          0
    vmsize:          64
    fileoff:         448
    filesize:        64
    maxprot:         7
    initprot:        7
    nsects:          2
    flags:           0
    Sections:
      - sectname:        __text
        segname:         __TEXT
        addr:            0x0000000000000000
        size:            1
        offset:          0x000001C0
        align:           0
        reloff:          0x00000000
        nreloc:          0
        flags:           0x80000400
        reserved1:       0x00000000
        reserved2:       0x00000000
        reserved3:       0x00000000
        content:         '90'
      - sectname:        __compact_unwind
        segname:         __LD
        addr:            0x0000000000000020
        size:            32
        offset:          0x000001E0
        align:           3
        reloff:          0x00000200
        nreloc:          1
        flags:           0x02000000
        reserved1:       0x00000000
        reserved2:       0x00000000
        reserved3:       0x00000000
        content:         '0000000000000000010000000000020200000000000000000000000000000000'
        relocations:
          - address:         0x00000000
            symbolnum:       0
            pcrel:           false
            length:          3
            extern:          true
            type:            0
            scattered:       false
            value:           0
  - cmd:             LC_SYMTAB
    cmdsize:         24
    symoff:          520
    nsyms:           1
    stroff:          552
    strsize:         8
LinkEditData:
  NameList:
    - n_strx:          2
      n_type:          0x0F
      n_sect:          1
      n_desc:          32
      n_value:         0
  StringTable:
    - ' '
    - _main

...
