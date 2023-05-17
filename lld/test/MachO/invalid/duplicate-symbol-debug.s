# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t-dup.o
# RUN: not %lld -dylib -arch arm64 -o /dev/null %t.o %t-dup.o 2>&1 | FileCheck %s -DFILE_1=%t.o -DFILE_2=%t-dup.o

# CHECK:      error: duplicate symbol: _foo
# CHECK-NEXT: >>> defined in duplicate-symbol-debug.s:20
# CHECK-NEXT: >>>            [[FILE_1]]
# CHECK-NEXT: >>> defined in duplicate-symbol-debug.s:20
# CHECK-NEXT: >>>            [[FILE_2]]

## Test case adapted from lld/test/ELF/Inputs/vs-diagnostics-duplicate2.s

.file 1 "" "duplicate-symbol-debug.s"

.text

.globl _foo
.loc 1 20
_foo:
  nop

.section __DWARF,__debug_abbrev,regular,debug
  .byte  1                  ; Abbreviation code
  .byte 17                  ; DW_TAG_compile_unit
  .byte  0                  ; DW_CHILDREN_no
  .byte 16                  ; DW_AT_stmt_list
  .byte 23                  ; DW_FORM_sec_offset
  .byte  0                  ; EOM(1)
  .byte  0                  ; EOM(2)
  .byte  0                  ; EOM(3)

.section __DWARF,__debug_info,regular,debug
  .long Lend0 - Lbegin0     ; Length of Unit
Lbegin0:
  .short 4                  ; DWARF version number
  .long  __debug_abbrev     ; Offset Into Abbrev. Section
  .byte  8                  ; Address Size (in bytes)
  .byte  1                  ; Abbrev [1] 0xb:0x1f DW_TAG_compile_unit
  .long  __debug_line       ; DW_AT_stmt_list
Lend0:
  .section __DWARF,__debug_line,regular,debug

