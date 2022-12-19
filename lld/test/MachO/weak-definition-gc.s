# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t

## This tests that if two input files define the same weak symbol, we only
## write it to the output once (...assuming both input files use
## .subsections_via_symbols).

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weak-sub.s -o %t/weak-sub.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/weak-nosub.s -o %t/weak-nosub.o

## Test that weak symbols are emitted just once with .subsections_via_symbols
# RUN: %lld -dylib -o %t/out.dylib %t/weak-sub.o %t/weak-sub.o
# RUN: llvm-otool -jtV %t/out.dylib | FileCheck --check-prefix=SUB %s
# RUN: %lld -dylib -o %t/out.dylib %t/weak-nosub.o %t/weak-sub.o
# RUN: llvm-otool -jtV %t/out.dylib | FileCheck --check-prefix=SUB %s
# RUN: %lld -dylib -o %t/out.dylib %t/weak-sub.o %t/weak-nosub.o
# RUN: llvm-otool -jtV %t/out.dylib | FileCheck --check-prefix=SUB %s
# SUB:      _foo
# SUB-NEXT: retq
# SUB-NOT:  retq
# SUB:      _bar
# SUB-NEXT: retq
# SUB-NOT:  retq

## We can even strip weak symbols without subsections_via_symbols as long
## as none of the weak symbols in a section are needed.
# RUN: %lld -dylib -o %t/out.dylib %t/weak-nosub.o %t/weak-nosub.o
# RUN: llvm-otool -jtV %t/out.dylib | FileCheck --check-prefix=SUB %s

## Test that omitted weak symbols don't add entries to the compact unwind table.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/weak-sub-lsda.s -o %t/weak-sub-lsda.o
# RUN: %lld -dylib -lc++ -o %t/out.dylib %t/weak-sub-lsda.o %t/weak-sub-lsda.o
# RUN: llvm-objdump --macho --unwind-info --syms %t/out.dylib | FileCheck %s --check-prefix=ONE-UNWIND
# RUN: %lld -dylib -lc++ -o %t/out.dylib %t/weak-sub.o %t/weak-sub-lsda.o
# RUN: llvm-objdump --macho --unwind-info --syms %t/out.dylib | FileCheck %s --check-prefix=NO-UNWIND
# RUN: yaml2obj %t/weak-sub-lsda-r.yaml -o %t/weak-sub-lsda-r.o
# RUN: %lld -dylib -lc++ -o %t/out.dylib %t/weak-sub.o %t/weak-sub-lsda-r.o
# RUN: llvm-objdump --macho --unwind-info --syms %t/out.dylib | FileCheck %s --check-prefix=NO-UNWIND

# ONE-UNWIND:      SYMBOL TABLE:
# ONE-UNWIND-DAG:  [[#%x,FOO:]]       w  F __TEXT,__text _foo
# ONE-UNWIND-NOT:                          __TEXT,__text _foo

# ONE-UNWIND:      Contents of __unwind_info section:
# ONE-UNWIND:        LSDA descriptors:
# ONE-UNWIND:           [0]: function offset=0x[[#%.8x,FOO]]
# ONE-UNWIND-NOT:       [1]:
# ONE-UNWIND:        Second level indices:
# ONE-UNWIND-DAG:      [0]: function offset=0x[[#%.8x,FOO]]
# ONE-UNWIND-NOT:      [1]:

# NO-UNWIND:      SYMBOL TABLE:
# NO-UNWIND-DAG:  [[#%x,FOO:]]       w  F __TEXT,__text _foo
# NO-UNWIND-NOT:                          __TEXT,__text _foo
# NO-UNWIND-NOT:  Contents of __unwind_info section:

## Test interaction with .alt_entry
## FIXME: ld64 manages to strip both one copy of _foo and _bar each.
##        We only manage this if we're lucky and the object files are in
##        the right order. We're happy to not crash at link time for now.
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/weak-sub-alt.s -o %t/weak-sub-alt.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/weak-sub-alt2.s -o %t/weak-sub-alt2.o
# RUN: %lld -dylib -o %t/out.dylib %t/weak-sub-alt.o %t/weak-sub-alt2.o
# RUN: %lld -dylib -o %t/out.dylib %t/weak-sub-alt2.o %t/weak-sub-alt.o

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/weak-aligned-1.s -o %t/weak-aligned-1.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/weak-aligned-2.s -o %t/weak-aligned-2.o
# RUN: %lld -o %t/out -lSystem %t/weak-aligned-1.o %t/weak-aligned-2.o
# RUN: llvm-objdump --syms --section=__const --full-contents %t/out | FileCheck --check-prefixes=ALIGN,ALIGN2 %s
# RUN: %lld -o %t/out -lSystem %t/weak-aligned-1.o %t/weak-aligned-2.o -dead_strip
# RUN: llvm-objdump --syms --section=__const --full-contents %t/out | FileCheck --check-prefixes=ALIGN,ALIGN3 %s
# ALIGN:       SYMBOL TABLE:
# ALIGN-DAG:   [[#%x, ADDR:]]       l     O __DATA_CONST,__const .hidden _weak1
# ALIGN2-DAG:  {{0*}}[[#ADDR+ 0x4]] l     O __DATA_CONST,__const .hidden _weak3
# ALIGN3-DAG:  {{0*}}[[#ADDR+ 0x4]] l     O __DATA_CONST,__const .hidden _weak2
# ALIGN2-DAG:  {{0*}}[[#ADDR+ 0x8]] l     O __DATA_CONST,__const .hidden _weak2
# ALIGN-DAG:   {{0*}}[[#ADDR+0x10]] g     O __DATA_CONST,__const _aligned
# ALIGN:       Contents of section __DATA_CONST,__const:
# ALIGN2-NEXT: {{0*}}[[#ADDR]]      11111111 33333333 22222222 00000000
# ALIGN3-NEXT: {{0*}}[[#ADDR]]      11111111 22222222 00000000 00000000
# ALIGN-NEXT:  {{0*}}[[#ADDR+0x10]] 81818181 81818181 82828282 82828282

# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/weak-def.s -o %t/weak-def.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin19.0.0 %t/strong-def.s -o %t/strong-def.o
# RUN: %lld -dylib -lc++ -o %t/weak-strong-mixed.dylib %t/weak-def.o %t/strong-def.o
# RUN: %lld -dylib -lc++ -o %t/strong-weak-mixed.dylib %t/strong-def.o %t/weak-def.o
## Check that omitted weak symbols are not adding their section and unwind stuff.

# RUN: llvm-otool -jtV %t/weak-strong-mixed.dylib | FileCheck --check-prefix=MIXED %s
# RUN: llvm-otool -jtV %t/strong-weak-mixed.dylib | FileCheck --check-prefix=MIXED %s
# MIXED: (__TEXT,__text) section
# MIXED-NEXT: _foo:
# MIXED-NEXT: {{.+}} 	33 33            	xorl	(%rbx), %esi
# MIXED-NEXT: {{.+}} 	33 33            	xorl	(%rbx), %esi
# MIXED-NEXT: {{.+}}	c3              	retq

# RUN: llvm-objdump --macho --syms --unwind-info %t/weak-strong-mixed.dylib | FileCheck --check-prefix=MIXED-UNWIND %s
# RUN: llvm-objdump --macho --syms --unwind-info %t/strong-weak-mixed.dylib | FileCheck --check-prefix=MIXED-UNWIND %s
# MIXED-UNWIND: g     F __TEXT,__text _foo
# MIXED-UNWIND-NOT: Contents of __unwind_info section:

#--- weak-sub.s
.globl _foo, _bar
.weak_definition _foo, _bar
_foo:
  retq
_bar:
  retq
.subsections_via_symbols

#--- weak-nosub.s
.globl _foo, _bar
.weak_definition _foo, _bar
_foo:
  retq
_bar:
  retq

#--- weak-sub-lsda.s
.section __TEXT,__text,regular,pure_instructions

.globl _foo
.weak_definition _foo
_foo:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_lsda 16, Lexception
  pushq %rbp
  .cfi_def_cfa_offset 128
  .cfi_offset %rbp, 48
  movq %rsp, %rbp
  .cfi_def_cfa_register %rbp
  popq %rbp
  retq
  .cfi_endproc

.section __TEXT,__gcc_except_tab
Lexception:
    .space 0x10

.subsections_via_symbols

#--- weak-sub-alt.s
.globl _foo, _bar
.weak_definition _foo
_foo:
  retq

# Alternative entry point to _foo (strong)
.alt_entry _bar
_bar:
  retq

.globl _main, _ref
_main:
  callq _ref
  callq _bar

.subsections_via_symbols

#--- weak-sub-alt2.s
.globl _foo, _bar
.weak_definition _foo
_foo:
  retq

# Alternative entry point to _foo (weak)
.weak_definition _bar
.alt_entry _bar
_bar:
  retq

.globl _ref
_ref:
  callq _bar

.subsections_via_symbols

#--- weak-aligned-1.s
.section __DATA,__const
.p2align 3
.globl _weak1
.weak_def_can_be_hidden _weak1
_weak1:
  .4byte 0x11111111

.globl _weak3
.weak_def_can_be_hidden _weak3
_weak3:
  .4byte 0x33333333

.subsections_via_symbols

#--- weak-aligned-2.s
# _weak1 and _weak3 are already in weak-aligned-1,
# so from _weak1-3 in this file only _weak2 is used.
# However, _aligned still has to stay aligned to a 16-byte boundary.
.section __DATA,__const
.p2align 3
.globl _weak1
.weak_def_can_be_hidden _weak1
_weak1:
  .4byte 0x11111111

.globl _weak2
.weak_def_can_be_hidden _weak2
_weak2:
  .4byte 0x22222222

.globl _weak3
.weak_def_can_be_hidden _weak3
_weak3:
  .4byte 0x33333333

.section __DATA,__const
.p2align 4
.globl _aligned
_aligned:
  .8byte 0x8181818181818181
  .8byte 0x8282828282828282

.section __TEXT,__text
.globl _main
_main:
  movl _weak1(%rip), %eax
  movl _weak2(%rip), %ebx
  movaps _aligned(%rip), %xmm0
  retq

.subsections_via_symbols

#--- weak-def.s
.section __TEXT,__text,regular,pure_instructions

.globl _foo
.weak_definition _foo
_foo:
  .cfi_startproc
  .cfi_personality 155, ___gxx_personality_v0
  .cfi_lsda 16, Lexception
  pushq %rbp
  .cfi_def_cfa_offset 128
  .cfi_offset %rbp, 48
  movq %rsp, %rbp
  .cfi_def_cfa_register %rbp
  popq %rbp
  retq
  .cfi_endproc

.section __TEXT,__gcc_except_tab
Lexception:
    .space 0x10

.subsections_via_symbols
#--- strong-def.s
.globl _foo, _bar

_foo:
  .4byte 0x33333333
  retq

.subsections_via_symbols

#--- weak-sub-lsda-r.yaml
## This was generated from compiling weak-sub-lsda.s above at rev a2404f11c77e
## and then running it through `ld -r`. This converts a number of unwind-related
## relocations from section- to symbol-based ones.
--- !mach-o
FileHeader:
  magic:           0xFEEDFACF
  cputype:         0x1000007
  cpusubtype:      0x3
  filetype:        0x1
  ncmds:           2
  sizeofcmds:      464
  flags:           0x2000
  reserved:        0x0
LoadCommands:
  - cmd:             LC_SEGMENT_64
    cmdsize:         392
    segname:         ''
    vmaddr:          0
    vmsize:          152
    fileoff:         528
    filesize:        152
    maxprot:         7
    initprot:        7
    nsects:          3
    flags:           0
    Sections:
      - sectname:        __text
        segname:         __TEXT
        addr:            0x0
        size:            6
        offset:          0x210
        align:           0
        reloff:          0x0
        nreloc:          0
        flags:           0x80000400
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         554889E55DC3
      - sectname:        __gcc_except_tab
        segname:         __TEXT
        addr:            0x6
        size:            32
        offset:          0x216
        align:           0
        reloff:          0x0
        nreloc:          0
        flags:           0x0
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         '0000000000000000000000000000000000000000000000000000000000000000'
      - sectname:        __eh_frame
        segname:         __TEXT
        addr:            0x28
        size:            80
        offset:          0x238
        align:           3
        reloff:          0x2A8
        nreloc:          7
        flags:           0x0
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         1C00000000000000017A504C5200017810079B0400000010100C0708900100002C00000004000000F8FFFFFFFFFFFFFF060000000000000008E7FFFFFFFFFFFFFF410E800111067A430D060000000000
        relocations:
          - address:         0x13
            symbolnum:       5
            pcrel:           true
            length:          2
            extern:          true
            type:            4
            scattered:       false
            value:           0
          - address:         0x24
            symbolnum:       2
            pcrel:           false
            length:          2
            extern:          true
            type:            5
            scattered:       false
            value:           0
          - address:         0x24
            symbolnum:       3
            pcrel:           false
            length:          2
            extern:          true
            type:            0
            scattered:       false
            value:           0
          - address:         0x28
            symbolnum:       3
            pcrel:           false
            length:          3
            extern:          true
            type:            5
            scattered:       false
            value:           0
          - address:         0x28
            symbolnum:       4
            pcrel:           false
            length:          3
            extern:          true
            type:            0
            scattered:       false
            value:           0
          - address:         0x39
            symbolnum:       3
            pcrel:           false
            length:          3
            extern:          true
            type:            5
            scattered:       false
            value:           0
          - address:         0x39
            symbolnum:       1
            pcrel:           false
            length:          3
            extern:          true
            type:            0
            scattered:       false
            value:           0
      - sectname:        __compact_unwind
        segname:         __LD
        addr:            0x78
        size:            32
        offset:          0x288
        align:           3
        reloff:          0x2E0
        nreloc:          4
        flags:           0x2000000
        reserved1:       0x0
        reserved2:       0x0
        reserved3:       0x0
        content:         '0000000000000000060000000000004100000000000000000000000000000000'
        relocations:
          - address:         0x0
            symbolnum:       4
            pcrel:           false
            length:          3
            extern:          true
            type:            0
            scattered:       false
            value:           0
          - address:         0x18
            symbolnum:       1
            pcrel:           false
            length:          3
            extern:          true
            type:            0
            scattered:       false
            value:           0
          - address:         0x10
            symbolnum:       5
            pcrel:           false
            length:          3
            extern:          true
            type:            0
            scattered:       false
            value:           0
          - address:         0x18
            symbolnum:       1
            pcrel:           false
            length:          3
            extern:          true
            type:            0
            scattered:       false
            value:           0
  - cmd:             LC_SYMTAB
    cmdsize:         24
    symoff:          768
    nsyms:           6
    stroff:          864
    strsize:         57
LinkEditData:
  NameList:
    - n_strx:          29
      n_type:          0xE
      n_sect:          2
      n_desc:          0
      n_value:         6
    - n_strx:          34
      n_type:          0xE
      n_sect:          2
      n_desc:          0
      n_value:         22
    - n_strx:          39
      n_type:          0xE
      n_sect:          3
      n_desc:          0
      n_value:         40
    - n_strx:          49
      n_type:          0xE
      n_sect:          3
      n_desc:          0
      n_value:         72
    - n_strx:          2
      n_type:          0xF
      n_sect:          1
      n_desc:          128
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
    - l001
    - l002
    - EH_Frame1
    - func.eh
...
