# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj --triple=wasm32-unknown-unknown -o %t/main.o %t/main.s
# RUN: llvm-mc -filetype=obj --triple=wasm32-unknown-unknown -o %t/extra.o %t/extra.s
# RUN: wasm-ld --relocatable -o %t/reloc.o %t/main.o %t/extra.o
# RUN: obj2yaml %t/reloc.o | FileCheck %s --check-prefix=RELOC
# RUN: wasm-ld --export-all %t/reloc.o -o %t/final.wasm
# RUN: obj2yaml %t/final.wasm | FileCheck %s --check-prefix=FINAL

# A non-STRINGS segment (such as an embedded-null string literal) must not have
# the STRINGS flag applied when coalesced with a STRINGS segment of the same
# name in a relocatable link. Otherwise, downstream links will treat the entire
# segment as mergeable strings, splitting the embedded-null string at '\0' and
# reordering/corrupting it.

#--- main.s
  .globl _start
_start:
  .functype _start () -> ()
  end_function

  .section .rodata.str,"S",@
  .asciz "hello"
  .asciz "apple"

#--- extra.s
  .section .rodata.str,"",@
  .globl str_with_null
str_with_null:
  .asciz "hello\000world"
  .size str_with_null, 12

# RELOC:      SegmentInfo:
# RELOC:        - Index:           0
# RELOC-NEXT:     Name:            .rodata.str
# RELOC-NEXT:     Alignment:       0
# RELOC-NEXT:     Flags:           [ STRINGS ]
# RELOC-NEXT:   - Index:           1
# RELOC-NEXT:     Name:            .rodata.str
# RELOC-NEXT:     Alignment:       0
# RELOC-NEXT:     Flags:           [ ]

# In the final binary, "hello\0world\0" must remain contiguous, and not be
# split apart by string-table merging inserting "apple" between "hello" and "world".
# FINAL:      - Type:            DATA
# FINAL-NEXT:   Segments:
# FINAL:          Content:         68656C6C6F00776F726C6400
