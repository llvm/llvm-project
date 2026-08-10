# REQUIRES: aarch64
## Check the order of local symbols: grouped by file, each group led by the file's STT symbol.
## Symbols converted to STB_LOCAL are placed at the end of the local part after a synthetic
## STT_FILE with an empty name, omitted if the output has no STT_FILE.

# RUN: rm -rf %t && split-file %s %t && cd %t
## -implicit-mapsyms omits the input $x/$d mapping symbols; only the linker's
## own thunk mapping symbols remain (THUNK below).
# RUN: llvm-mc -filetype=obj -triple=aarch64 -implicit-mapsyms a.s -o a.o
# RUN: llvm-mc -filetype=obj -triple=aarch64 -implicit-mapsyms b.s -o b.o
# RUN: mkdir c && llvm-mc -filetype=obj -triple=aarch64 -implicit-mapsyms c.s -o c/a.o

## Only b.c has an STT_FILE. a.o's a_local has no leading STT_FILE; the hidden
## a_hidden follows the synthetic STT_FILE. --emit-relocs adds STT_SECTION
## symbols to their file's group.
# RUN: ld.lld --emit-relocs a.o b.o -o ab
# RUN: llvm-readelf -s ab | FileCheck %s

# CHECK:      NOTYPE  LOCAL  DEFAULT [[#]] a_local
# CHECK-NEXT: SECTION LOCAL  DEFAULT [[#]] .text
# CHECK-NEXT: FILE    LOCAL  DEFAULT   ABS b.c
# CHECK-NEXT: NOTYPE  LOCAL  DEFAULT [[#]] b_local
# CHECK-NEXT: SECTION LOCAL  DEFAULT [[#]] .data
# CHECK-NEXT: SECTION LOCAL  DEFAULT [[#]] .comment
# CHECK-NEXT: FILE    LOCAL  DEFAULT   ABS{{ $}}
# CHECK-NEXT: NOTYPE  LOCAL  HIDDEN  [[#]] a_hidden
# CHECK-NEXT: NOTYPE  GLOBAL DEFAULT [[#]] a_localized
# CHECK-NEXT: NOTYPE  GLOBAL DEFAULT [[#]] a_exported
# CHECK-NEXT: NOTYPE  GLOBAL DEFAULT [[#]] b_localized
# CHECK-NEXT: NOTYPE  GLOBAL DEFAULT [[#]] b_exported

## For -r, synthesize STT_FILE for a.o, which lacks one.
# RUN: ld.lld -r a.o b.o -o ab.o
# RUN: llvm-readelf -s ab.o | FileCheck %s --check-prefix=RO

# RO:      FILE    LOCAL  DEFAULT   ABS a.o
# RO-NEXT: NOTYPE  LOCAL  DEFAULT [[#]] a_local
# RO-NEXT: SECTION LOCAL  DEFAULT [[#]] .text
# RO-NEXT: FILE    LOCAL  DEFAULT   ABS b.c
# RO-NEXT: NOTYPE  LOCAL  DEFAULT [[#]] b_local
# RO-NEXT: SECTION LOCAL  DEFAULT [[#]] .data
# RO-NEXT: NOTYPE  GLOBAL DEFAULT [[#]] a_localized
# RO-NEXT: NOTYPE  GLOBAL HIDDEN  [[#]] a_hidden
# RO-NEXT: NOTYPE  GLOBAL DEFAULT [[#]] a_exported
# RO-NEXT: NOTYPE  GLOBAL DEFAULT [[#]] b_localized
# RO-NEXT: NOTYPE  GLOBAL DEFAULT [[#]] b_exported

## Synthesize STT_FILE named after its basename, not its path, for local determinism (#47367).
# RUN: ld.lld -r ab.o c/a.o -o abc.o
# RUN: llvm-readelf -s abc.o | FileCheck %s --check-prefix=MULTI

# MULTI:      FILE    LOCAL  DEFAULT   ABS a.o
# MULTI-NEXT: NOTYPE  LOCAL  DEFAULT [[#]] a_local
# MULTI-NEXT: FILE    LOCAL  DEFAULT   ABS b.c
# MULTI-NEXT: NOTYPE  LOCAL  DEFAULT [[#]] b_local
# MULTI-NEXT: SECTION LOCAL  DEFAULT [[#]] .text
# MULTI-NEXT: SECTION LOCAL  DEFAULT [[#]] .data
# MULTI-NEXT: FILE    LOCAL  DEFAULT   ABS a.o
# MULTI-NEXT: NOTYPE  LOCAL  DEFAULT [[#]] far

## Symbols localized by a version script follow a synthetic STT_FILE with empty name.
# RUN: ld.lld -shared --version-script=ver ab.o -o ab.so
# RUN: llvm-readelf -s ab.so | FileCheck %s --check-prefix=VER

# VER:      FILE    LOCAL  DEFAULT   ABS a.o
# VER-NEXT: NOTYPE  LOCAL  DEFAULT [[#]] a_local
# VER-NEXT: FILE    LOCAL  DEFAULT   ABS b.c
# VER-NEXT: NOTYPE  LOCAL  DEFAULT [[#]] b_local
# VER-NEXT: FILE    LOCAL  DEFAULT   ABS{{ $}}
# VER-NEXT: NOTYPE  LOCAL  DEFAULT [[#]] a_localized
# VER-NEXT: NOTYPE  LOCAL  HIDDEN  [[#]] a_hidden
# VER-NEXT: NOTYPE  LOCAL  DEFAULT [[#]] b_localized
# VER-NEXT: NOTYPE  LOCAL  HIDDEN  [[#]] _DYNAMIC
# VER-NEXT: NOTYPE  GLOBAL DEFAULT [[#]] a_exported
# VER-NEXT: NOTYPE  GLOBAL DEFAULT [[#]] b_exported

## A range-extension thunk (and its $x/$d) is added by
## finalizeAddressDependentContent and parented to the internal file, which has
## no STT_FILE. Like the demoted c_hidden it cannot be attributed to a file, so
## it follows the synthetic STT_FILE.
# RUN: ld.lld c/a.o b.o -T lds -o cb
# RUN: llvm-readelf -s cb | FileCheck %s --check-prefix=THUNK

# THUNK:      NOTYPE  LOCAL  DEFAULT [[#]] far
# THUNK-NEXT: FILE    LOCAL  DEFAULT   ABS b.c
# THUNK-NEXT: NOTYPE  LOCAL  DEFAULT [[#]] b_local
# THUNK-NEXT: FILE    LOCAL  DEFAULT   ABS{{ $}}
# THUNK-NEXT: NOTYPE  LOCAL  HIDDEN  [[#]] c_hidden
# THUNK-NEXT: FUNC    LOCAL  DEFAULT [[#]] __AArch64AbsLongThunk_{{.*}}
# THUNK-NEXT: NOTYPE  LOCAL  DEFAULT [[#]] $x
# THUNK-NEXT: NOTYPE  LOCAL  DEFAULT [[#]] $d
# THUNK-NEXT: NOTYPE  GLOBAL DEFAULT [[#]] _start

## --discard-all discards STT_FILE symbols as well. With no STT_FILE in the
## output, no synthetic STT_FILE is added.
# RUN: ld.lld -shared --discard-all --version-script=ver ab.o -o ab2.so
# RUN: llvm-readelf -s ab2.so | FileCheck %s --check-prefix=DISCARD --implicit-check-not=FILE

# DISCARD:      NOTYPE  LOCAL  DEFAULT [[#]] a_localized
# DISCARD-NEXT: NOTYPE  LOCAL  HIDDEN  [[#]] a_hidden
# DISCARD-NEXT: NOTYPE  LOCAL  DEFAULT [[#]] b_localized
# DISCARD-NEXT: NOTYPE  LOCAL  HIDDEN  [[#]] _DYNAMIC
# DISCARD-NEXT: NOTYPE  GLOBAL DEFAULT [[#]] a_exported
# DISCARD-NEXT: NOTYPE  GLOBAL DEFAULT [[#]] b_exported

#--- a.s
a_local:
.globl a_localized
a_localized:
.globl a_hidden
.hidden a_hidden
a_hidden:
.globl a_exported
a_exported:
  nop

#--- b.s
.file "b.c"
.data
b_local:
.globl b_localized
b_localized:
.globl b_exported
b_exported:
  .byte 0

#--- c.s
.globl _start, c_hidden
.hidden c_hidden
_start:
  bl far
c_hidden:
  ret
.section .far,"ax"
far:
  ret

#--- lds
SECTIONS {
  .text 0x10000    : { *(.text) }
  .far  0x10000000 : { *(.far) }
}

#--- ver
v1 { global: *_exported; local: *; };
