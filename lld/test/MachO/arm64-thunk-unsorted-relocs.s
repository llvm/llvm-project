# REQUIRES: aarch64

## Test that assertion-enabled lld does not crash when handling ld -r inputs
## that need branch extension.

# RUN: rm -rf %t; split-file %s %t
# RUN: yaml2obj %t/main.yaml -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %t/far.s -o %t/far.o
# RUN: %lld -arch arm64 -dylib -o %t/out %t/main.o %t/far.o
# RUN: llvm-nm -n %t/out | FileCheck %s

## Make sure we correctly sort the relocs so they satisfy the branch extension
## algorithm's expectation.
# CHECK:      _a.thunk.0
# CHECK-NEXT: _b.thunk.0

#--- far.s
.globl _a, _b
.space 0x8000000
_a:
  ret
_b:
  ret

#--- main.yaml
--- !mach-o
IsLittleEndian: true
FileHeader:
  magic:      0xFEEDFACF
  cputype:    0x100000C
  cpusubtype: 0x0
  filetype:   0x1
  ncmds:      2
  sizeofcmds: 176
  flags:      0x0
  reserved:   0x0
LoadCommands:
  - cmd:      LC_SEGMENT_64
    cmdsize:  152
    segname:  ''
    vmaddr:   0
    vmsize:   8
    fileoff:  208
    filesize: 8
    maxprot:  7
    initprot: 7
    nsects:   1
    flags:    0
    Sections:
      - sectname:  __text
        segname:   __TEXT
        addr:      0x0
        size:      8
        offset:    208
        align:     2
        reloff:    216
        nreloc:    2
        flags:     0x80000400
        reserved1: 0x0
        reserved2: 0x0
        reserved3: 0x0
        ## bl _a; bl _b, with their relocations in ascending order of address.
        content:   '0000009400000094'
        relocations:
          - { address: 0x0, symbolnum: 0, pcrel: true, length: 2, extern: true, type: 2, scattered: false, value: 0 }
          - { address: 0x4, symbolnum: 1, pcrel: true, length: 2, extern: true, type: 2, scattered: false, value: 0 }
  - cmd:     LC_SYMTAB
    cmdsize: 24
    symoff:  232
    nsyms:   2
    stroff:  264
    strsize: 7
LinkEditData:
  NameList:
    - { n_strx: 1, n_type: 0x1, n_sect: 0, n_desc: 0, n_value: 0 }
    - { n_strx: 4, n_type: 0x1, n_sect: 0, n_desc: 0, n_value: 0 }
  StringTable: [ '', _a, _b ]
...
