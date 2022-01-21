## Tests that lld correctly resolves the custom personality referenced by objc code in an archive.

# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-iossimulator %t/defined.s -o %t/defined.o
# RUN: yaml2obj %t/combined.yaml > %t/combined.o
# RUN: llvm-ar r %t/pack.a %t/defined.o %t/combined.o
# RUN: %lld -dylib -arch x86_64 -platform_version ios-simulator 12.0.0 15.0 -ObjC %t/pack.a -o %t/a.dylib
# RUN: llvm-objdump --macho --syms %t/a.dylib | FileCheck %s
# RUN: %lld -dylib -arch x86_64 -platform_version ios-simulator 12.0.0 15.0 -ObjC --start-lib %t/defined.o %t/combined.o --end-lib -o %t/a.dylib
# RUN: llvm-objdump --macho --syms %t/a.dylib | FileCheck %s

# CHECK: SYMBOL TABLE:
# CHECK: {{.*}}  l     F __TEXT,__text _my_personality


#--- defined.s
.private_extern _my_personality

.text
.no_dead_strip _my_personality
_my_personality:
.cfi_startproc
.cfi_def_cfa_offset 16
.cfi_endproc
nop
.subsections_via_symbols

## combined.yaml is produced from combined.o below:
## lvm-mc -filetype=obj -triple=x86_64-apple-iossimulator %t/objc.s -o %t/objc.o
## ld -r -o combined.o defined.o objc.o
## // objc.s:
## .section __TEXT,__text
## .global _OBJC_CLASS_$_MyTest
## .no_dead_strip _OBJC_CLASS_$_MyTest
## _OBJC_CLASS_$_MyTest:
##  .cfi_startproc
##  .cfi_personality 155, _my_personality
##  .cfi_def_cfa_offset 16
##  ret
##  .cfi_endproc
##
##  ret
## .subsections_via_symbols

#--- combined.yaml
--- !mach-o
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x1000007
  cpusubtype:      0x3
  filetype:        0x1
  ncmds:           3
  sizeofcmds:      352
  flags:           0x2000
  reserved:        0x0
LoadCommands:
  - cmd:             LC_SEGMENT_64
    cmdsize:         312
    segname:         ''
    vmaddr:          0
    vmsize:          152
    fileoff:         416
    filesize:        152
    maxprot:         7
    initprot:        7
    nsects:          3
    flags:           0
    Sections:
      - sectname:        __text
        segname:         __TEXT
        addr:            0x0
        size:            3
        offset:          0x1A0
        align:           0
        reloff:          0x0
        nreloc:          0
        flags:           0x80000400
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         90C3C3
      - sectname:        __eh_frame
        segname:         __TEXT
        addr:            0x8
        size:            80
        offset:          0x1A8
        align:           3
        reloff:          0x238
        nreloc:          5
        flags:           0x0
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         1400000000000000017A520001781001100C0708900100001800000000000000017A505200017810069B04000000100C070890011800000004000000F8FFFFFFFFFFFFFF0100000000000000000E1000
        relocations:
          - address:         0x2A
            symbolnum:       0
            pcrel:           true
            length:          2
            extern:          true
            type:            4
            scattered:       false
            value:           0
          - address:         0x38
            symbolnum:       2
            pcrel:           false
            length:          2
            extern:          true
            type:            5
            scattered:       false
            value:           0
          - address:         0x38
            symbolnum:       3
            pcrel:           false
            length:          2
            extern:          true
            type:            0
            scattered:       false
            value:           0
          - address:         0x3C
            symbolnum:       3
            pcrel:           false
            length:          3
            extern:          true
            type:            5
            scattered:       false
            value:           0
          - address:         0x3C
            symbolnum:       4
            pcrel:           false
            length:          3
            extern:          true
            type:            0
            scattered:       false
            value:           0
      - sectname:        __compact_unwind
        segname:         __LD
        addr:            0x58
        size:            64
        offset:          0x1F8
        align:           3
        reloff:          0x260
        nreloc:          3
        flags:           0x2000000
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         '00000000000000000100000000000202000000000000000000000000000000000000000000000000020000000000020200000000000000000000000000000000'
        relocations:
          - address:         0x0
            symbolnum:       0
            pcrel:           false
            length:          3
            extern:          true
            type:            0
            scattered:       false
            value:           0
          - address:         0x20
            symbolnum:       4
            pcrel:           false
            length:          3
            extern:          true
            type:            0
            scattered:       false
            value:           0
          - address:         0x30
            symbolnum:       0
            pcrel:           false
            length:          3
            extern:          true
            type:            0
            scattered:       false
            value:           0
  - cmd:             LC_SYMTAB
    cmdsize:         24
    symoff:          632
    nsyms:           5
    stroff:          712
    strsize:         64
  - cmd:             LC_DATA_IN_CODE
    cmdsize:         16
    dataoff:         632
    datasize:        0
LinkEditData:
  NameList:
    - n_strx:          23
      n_type:          0x1E
      n_sect:          1
      n_desc:          32
      n_value:         0
    - n_strx:          39
      n_type:          0xE
      n_sect:          2
      n_desc:          0
      n_value:         8
    - n_strx:          39
      n_type:          0xE
      n_sect:          2
      n_desc:          0
      n_value:         32
    - n_strx:          49
      n_type:          0xE
      n_sect:          2
      n_desc:          0
      n_value:         60
    - n_strx:          2
      n_type:          0xF
      n_sect:          1
      n_desc:          32
      n_value:         1
  StringTable:
    - ' '
    - '_OBJC_CLASS_$_MyTest'
    - _my_personality
    - EH_Frame1
    - func.eh
    - ''
    - ''
    - ''
    - ''
    - ''
    - ''
    - ''
...
