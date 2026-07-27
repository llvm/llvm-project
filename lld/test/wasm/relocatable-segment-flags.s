# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj --triple=wasm32-unknown-unknown -o %t/main.o %t/main.s
# RUN: llvm-mc -filetype=obj --triple=wasm32-unknown-unknown -o %t/extra.o %t/extra.s
# RUN: wasm-ld --relocatable -o %t/reloc.o %t/main.o %t/extra.o
# RUN: obj2yaml %t/reloc.o | FileCheck %s

# --relocatable must preserve per-segment linking flags (RETAIN, STRINGS). The plain
# "retained" chunk in extra.s coalesces with the retained one in main.o, so RETAIN must
# survive the flag union rather than be overwritten by the flag-less chunk.

#--- main.s
  .globl  _start
_start:
  .functype _start () -> ()
  end_function

  .section retained,"R",@
  .asciz  "keep"

  .section .rodata.str,"S",@
  .asciz  "merge"

  .section plain,"",@
  .asciz  "drop"

#--- extra.s
  .section retained,"",@
  .asciz  "more"

# CHECK:      SegmentInfo:
# CHECK:          Name:            .rodata.str
# CHECK-NEXT:     Alignment:       0
# CHECK-NEXT:     Flags:           [ STRINGS ]
# CHECK:          Name:            retained
# CHECK-NEXT:     Alignment:       0
# CHECK-NEXT:     Flags:           [ RETAIN ]
# CHECK:          Name:            plain
# CHECK-NEXT:     Alignment:       0
# CHECK-NEXT:     Flags:           [ ]
