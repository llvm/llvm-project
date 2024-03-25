## Tests that lld-macho can handle the case where personality symbols with the same name
## are both from a dylib and locally defined in an object file.

# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -emit-compact-unwind-non-canonical=true -filetype=obj  -triple=x86_64-apple-darwin19.0.0 %t/user_2.s -o %t/user_2.o
# RUN: llvm-mc -emit-compact-unwind-non-canonical=true -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/user_3.s -o %t/user_3.o

## Test case for linking 3+ personaltiies (all globals) without crashing because all the non-canonical are DWARFs
# RUN: llvm-mc -emit-compact-unwind-non-canonical=false -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/user_4.s -o %t/user_4.o
	
# RUN: yaml2obj %t/combined.yaml > %t/combined.o

## Pre-condition: check that ___gxx_personality_v0 really is locally defined in combined.o before we proceed.
# RUN: llvm-nm %t/combined.o | grep '___gxx_personality_v0'  | FileCheck %s --check-prefix=PRECHECK
# PRECHECK: {{.*}} t ___gxx_personality_v0
# PRECHECK-NOT: {{.*}} U ___gxx_personality_v0
# PRECHECK-NOT: {{.*}} T ___gxx_personality_v0

## check that we can link with 4 personalities without crashing:
## ___gxx_personality_v0 (libc++.tbd), ___gxx_personality_v0(local), _personality_1, and _personality_2
# RUN: %lld -lSystem -lc++ %t/user_2.o %t/combined.o -o %t/a.out
## ___gxx_personality_v0 (global), ___gxx_personality_v0(libc++.tbd), _personality_1, and _personality_2
# RUN: %lld -lSystem -lc++ %t/user_3.o %t/user_2.o -o %t/b.out
## ___gxx_personality_v0 (global), ___gxx_personality_v0(local), _personality_1, and _personality_2
# RUN: %lld -lSystem -dylib %t/user_3.o %t/combined.o %t/user_2.o -o %t/c.out

## check that we can link with 3+ personalities without crashing because
## non-canonical personalities are dwarf.
## This has ___gxx_personality_v0(global), ___gxx_personality_v0(local), and _personality_{1,2,3,4}
## Only the ___gxx_personality_v0(global) should have compact-unwind. The rest should have DWARFs.
# RUN: %lld -lSystem -lc++ %t/user_4.o %t/combined.o -o %t/d.out

## Postlink checks.
# RUN: llvm-nm %t/a.out | FileCheck %s --check-prefix=POSTCHECK
# POSTCHECK: {{.*}} U ___gxx_personality_v0
# POSTCHECK: {{.*}} t ___gxx_personality_v0

# RUN: llvm-objdump --macho --unwind-info --syms --indirect-symbols --bind %t/a.out | FileCheck %s --check-prefixes=A,CHECK -D#%x,OFF=0x100000000
# RUN: llvm-objdump --macho --unwind-info --syms --indirect-symbols --bind %t/b.out | FileCheck %s --check-prefixes=BC,CHECK -D#%x,OFF=0x100000000
# RUN: llvm-objdump --macho --unwind-info --syms --indirect-symbols --bind %t/c.out | FileCheck %s --check-prefixes=BC,C,CHECK -D#%x,OFF=0

# RUN: llvm-objdump --macho --indirect-symbols --unwind-info --bind %t/d.out | FileCheck %s --check-prefixes=D -D#%x,OFF=0x100000000

# A:      Indirect symbols for (__DATA_CONST,__got)
# A-NEXT: address                    index name
# A:      0x[[#%x,GXX_PERSONALITY_LO:]] [[#]] ___gxx_personality_v0
# A:      0x[[#%x,GXX_PERSONALITY_HI:]] [[#]] ___gxx_personality_v0
# A:      0x[[#%x,PERSONALITY_1:]]  LOCAL
# A:      0x[[#%x,PERSONALITY_2:]]  LOCAL

# BC:      Indirect symbols for (__DATA_CONST,__got)
# BC-NEXT: address                    index name
# C:       0x[[#%x,GXX_PERSONALITY_HI:]] LOCAL
# BC:      0x[[#%x,GXX_PERSONALITY_LO:]] LOCAL
# BC:      0x[[#%x,PERSONALITY_1:]]      LOCAL
# BC:      0x[[#%x,PERSONALITY_2:]]      LOCAL

# CHECK:        Personality functions: (count = 3)
# CHECK-DAG:     personality[{{[0-9]+}}]: 0x{{0*}}[[#GXX_PERSONALITY_LO-OFF]]
# CHECK-DAG:     personality[{{[0-9]+}}]: 0x{{0*}}[[#PERSONALITY_1-OFF]]
# CHECK-DAG:     personality[{{[0-9]+}}]: 0x{{0*}}[[#PERSONALITY_2-OFF]]

# A: Bind table
# A-NEXT: segment  section          address      type       addend dylib            symbol
# A-NEXT: __DATA_CONST __got        0x[[#GXX_PERSONALITY_LO-0]] pointer         0 libc++abi        ___gxx_personality_v0


# D:      Indirect symbols for (__DATA_CONST,__got)
# D-NEXT: address                    index name
# D:      0x[[#%x,GXX_PERSONALITY_HI:]] [[#]] ___gxx_personality_v0
# D:      0x[[#%x,PERSONALITY_1:]] [[#]] _personality_1
# D:      0x[[#%x,PERSONALITY_2:]] [[#]] _personality_2
# D:      0x[[#%x,PERSONALITY_3:]] [[#]] _personality_3
# D:      0x[[#%x,PERSONALITY_4:]] [[#]] _personality_4
# D:      0x[[#%x,GXX_PERSONALITY_LO:]] [[#]] ___gxx_personality_v0

# D: Contents of __unwind_info section:
# D:  Personality functions: (count = 1)
     personality[1]: 0x{{0*}}[[#GXX_PERSONALITY_HI-OFF]]
 
# D: Bind table:
# D: segment  section          address    type       addend dylib            symbol
# D: __DATA_CONST __got        0x[[#GXX_PERSONALITY_HI-0]] pointer         0 libc++abi        ___gxx_personality_v0


## Error cases.
## Check that dylib symbols are picked (which means without libc++, we'd get an undefined symbol error.
# RUN:  not %lld -lSystem %t/user_2.o %t/combined.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=ERRORCHECK
# ERRORCHECK: {{.*}} undefined symbol: ___gxx_personality_v0

#--- user_4.s
.globl _main, _personality_1, _personality_2, _personality_3, _personality_4

.text

_baz1:
  .cfi_startproc
  .cfi_personality 155, _personality_1
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_baz2:
  .cfi_startproc
  .cfi_personality 155, _personality_2
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_baz3:
  .cfi_startproc
  .cfi_personality 155, _personality_3
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc


_baz4:
  .cfi_startproc
  .cfi_personality 155, _personality_4
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc
  
_main:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_personality_1:
  retq
_personality_2:
  retq
_personality_3:
  retq
_personality_4:
  retq
	
#--- user_3.s
.globl _baz3
.private_extern ___gxx_personality_v0

.text
_baz3:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

.text
.no_dead_strip ___gxx_personality_v0
___gxx_personality_v0:
  nop

.subsections_via_symbols


#--- user_2.s
.globl _main, _personality_1, _personality_2

.text

_bar:
  .cfi_startproc
  .cfi_personality 155, _personality_1
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_baz:
  .cfi_startproc
  .cfi_personality 155, _personality_2
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_main:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_def_cfa_offset 16
  retq
  .cfi_endproc

_personality_1:
  retq
_personality_2:
  retq

## This yaml was created from the combined.o object file described in this comment:
## https://reviews.llvm.org/D107533#2935217
#--- combined.yaml
--- !mach-o
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x1000007
  cpusubtype:      0x3
  filetype:        0x1
  ncmds:           4
  sizeofcmds:      384
  flags:           0x2000
  reserved:        0x0
LoadCommands:
  - cmd:             LC_SEGMENT_64
    cmdsize:         312
    segname:         ''
    vmaddr:          0
    vmsize:          152
    fileoff:         448
    filesize:        152
    maxprot:         7
    initprot:        7
    nsects:          3
    flags:           0
    Sections:
      - sectname:        __text
        segname:         __TEXT
        addr:            0x0
        size:            5
        offset:          0x1C0
        align:           2
        reloff:          0x0
        nreloc:          0
        flags:           0x80000400
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         90909090C3
      - sectname:        __eh_frame
        segname:         __TEXT
        addr:            0x8
        size:            80
        offset:          0x1C8
        align:           3
        reloff:          0x258
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
        offset:          0x218
        align:           3
        reloff:          0x280
        nreloc:          3
        flags:           0x2000000
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         '00000000000000000100000000000202000000000000000000000000000000000000000000000000010000000000020200000000000000000000000000000000'
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
    symoff:          664
    nsyms:           5
    stroff:          744
    strsize:         48
  - cmd:             LC_BUILD_VERSION
    cmdsize:         32
    platform:        1
    minos:           659200
    sdk:             720896
    ntools:          1
    Tools:
      - tool:            3
        version:         39913472
  - cmd:             LC_DATA_IN_CODE
    cmdsize:         16
    dataoff:         664
    datasize:        0
LinkEditData:
  NameList:
    - n_strx:          7
      n_type:          0x1E
      n_sect:          1
      n_desc:          32
      n_value:         0
    - n_strx:          29
      n_type:          0xE
      n_sect:          2
      n_desc:          0
      n_value:         8
    - n_strx:          29
      n_type:          0xE
      n_sect:          2
      n_desc:          0
      n_value:         32
    - n_strx:          39
      n_type:          0xE
      n_sect:          2
      n_desc:          0
      n_value:         60
    - n_strx:          2
      n_type:          0xF
      n_sect:          1
      n_desc:          0
      n_value:         4
  StringTable:
    - ' '
    - _foo
    - ___gxx_personality_v0
    - EH_Frame1
    - func.eh
    - ''
...
