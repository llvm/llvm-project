# REQUIRES: x86

## Test that we correctly handle symbol relocations in the compact unwind
## section.

## llvm-mc does not emit such relocations for compact unwind, but `ld -r` does.
## As such, these yaml files were from an object file produced with 'ld -r',
## specifically:
##
##  // foo.s
## .globl _foo
## .text
## .p2align 2
## _foo:
##   .cfi_startproc
##   .cfi_personality 155, ___gxx_personality_v0
##   .cfi_lsda 16, _exception0
##   .cfi_def_cfa_offset 16
##   ret
##   .cfi_endproc
##
## .section __TEXT,__gcc_except_tab
## _exception0:
##   .space 0
##
## llvm-mc -filetype=obj -triple=x86_64-apple-macos10.15 -o foo1.o foo.s
## ld -r -o foo.o foo1.o

# RUN: rm -rf %t; mkdir -p %t
# RUN: yaml2obj %s -o %t/foo.o
# RUN: %lld -dylib -lc++ %t/foo.o -o %t/foo
# RUN: llvm-objdump --macho --syms --unwind-info %t/foo | FileCheck %s

# CHECK:      SYMBOL TABLE:
# CHECK-DAG:  [[#%x,FOO:]]         g     F __TEXT,__text _foo
# CHECK-DAG:  [[#%x,EXCEPT0:]]     l     O __TEXT,__gcc_except_tab _exception0

# CHECK:      LSDA descriptors:
# CHECK-NEXT: [0]: function offset=0x[[#%.8x,FOO]], LSDA offset=0x[[#%.8x,EXCEPT0]]

--- !mach-o
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x1000007
  cpusubtype:      0x3
  filetype:        0x1
  ncmds:           2
  sizeofcmds:      464
  flags:           0x0
  reserved:        0x0
LoadCommands:
  - cmd:             LC_SEGMENT_64
    cmdsize:         392
    segname:         ''
    vmaddr:          0
    vmsize:          112
    fileoff:         528
    filesize:        112
    maxprot:         7
    initprot:        7
    nsects:          4
    flags:           0
    Sections:
      - sectname:        __text
        segname:         __TEXT
        addr:            0x0
        size:            1
        offset:          0x210
        align:           2
        reloff:          0x0
        nreloc:          0
        flags:           0x80000400
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         C3
      - sectname:        __gcc_except_tab
        segname:         __TEXT
        addr:            0x1
        size:            0
        offset:          0x211
        align:           0
        reloff:          0x0
        nreloc:          0
        flags:           0x0
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         ''
      - sectname:        __eh_frame
        segname:         __TEXT
        addr:            0x8
        size:            72
        offset:          0x218
        align:           3
        reloff:          0x280
        nreloc:          7
        flags:           0x0
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         1C00000000000000017A504C5200017810079B0400000010100C0708900100002400000004000000F8FFFFFFFFFFFFFF010000000000000008E7FFFFFFFFFFFFFF0E100000000000
        relocations:
          - address:         0x13
            symbolnum:       4
            pcrel:           true
            length:          2
            extern:          true
            type:            4
            scattered:       false
            value:           0
          - address:         0x24
            symbolnum:       1
            pcrel:           false
            length:          2
            extern:          true
            type:            5
            scattered:       false
            value:           0
          - address:         0x24
            symbolnum:       2
            pcrel:           false
            length:          2
            extern:          true
            type:            0
            scattered:       false
            value:           0
          - address:         0x28
            symbolnum:       2
            pcrel:           false
            length:          3
            extern:          true
            type:            5
            scattered:       false
            value:           0
          - address:         0x28
            symbolnum:       3
            pcrel:           false
            length:          3
            extern:          true
            type:            0
            scattered:       false
            value:           0
          - address:         0x39
            symbolnum:       2
            pcrel:           false
            length:          3
            extern:          true
            type:            5
            scattered:       false
            value:           0
          - address:         0x39
            symbolnum:       0
            pcrel:           false
            length:          3
            extern:          true
            type:            0
            scattered:       false
            value:           0
      - sectname:        __compact_unwind
        segname:         __LD
        addr:            0x50
        size:            32
        offset:          0x260
        align:           3
        reloff:          0x2B8
        nreloc:          4
        flags:           0x2000000
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         '0000000000000000010000000000024200000000000000000000000000000000'
        relocations:
          - address:         0x0
            symbolnum:       3
            pcrel:           false
            length:          3
            extern:          true
            type:            0
            scattered:       false
            value:           0
          - address:         0x18
            symbolnum:       0
            pcrel:           false
            length:          3
            extern:          true
            type:            0
            scattered:       false
            value:           0
          - address:         0x10
            symbolnum:       4
            pcrel:           false
            length:          3
            extern:          true
            type:            0
            scattered:       false
            value:           0
          - address:         0x18
            symbolnum:       0
            pcrel:           false
            length:          3
            extern:          true
            type:            0
            scattered:       false
            value:           0
  - cmd:             LC_SYMTAB
    cmdsize:         24
    symoff:          728
    nsyms:           5
    stroff:          808
    strsize:         57
LinkEditData:
  NameList:
    - n_strx:          29
      n_type:          0xE
      n_sect:          2
      n_desc:          32
      n_value:         1
    - n_strx:          41
      n_type:          0xE
      n_sect:          3
      n_desc:          0
      n_value:         8
    - n_strx:          51
      n_type:          0xE
      n_sect:          3
      n_desc:          0
      n_value:         40
    - n_strx:          2
      n_type:          0xF
      n_sect:          1
      n_desc:          32
      n_value:         0
    - n_strx:          7
      n_type:          0x1
      n_sect:          0
      n_desc:          0
      n_value:         0
  StringTable:
    - ' '
    - _foo
    - ___gxx_personality_v0
    - _exception0
    - EH_Frame1
    - func.eh
...
