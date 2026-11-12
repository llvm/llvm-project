# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-linux-musl %s -o %t.o
# RUN: echo 'SECTIONS { .text 0x20000 : { *(.text) } .text.foo 0x1020000 : { *(.text.foo) } }' > %t.lds
# RUN: ld.lld -T %t.lds %t.o -o %t
# RUN: llvm-objdump -d %t 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-NONPIC %s
# RUN: ld.lld -T %t.lds --pie %t.o -o %t
# RUN: llvm-objdump -d %t 2>&1 | FileCheck --check-prefixes=CHECK,CHECK-PIC %s

## Packets with pc-relative relocations are more interesting because
## the offset must be relative to the start of the source, destination
## packets and not necessarily the instruction word containing the jump/call.

# CHECK:  Disassembly of section .text:

# CHECK:      00020000 <myfn_b>:
# CHECK:      { jumpr r31 }
    .globl myfn_b
    .type  myfn_b, @function
myfn_b:
    jumpr r31
    .size  myfn_b, .-myfn_b

# CHECK:      00020004 <main>:
    .globl main
    .type  main, @function
main:
    { r0 = #0
      call myfn_a }
# CHECK:        { call 0x20054 <__hexagon_thunk_myfn_a_from_.text.thunk>
# CHECK-NEXT:     r0 = #0x0 }
    call myfn_a
# CHECK:        call 0x20054 <__hexagon_thunk_myfn_a_from_.text.thunk>
    call myfn_b
# CHECK-NEXT:   call 0x20000 <myfn_b>

    { r2 = add(r0, r1)
      if (p0) call #myfn_b
      if (!p0) call #myfn_a }
# CHECK-NEXT:   { if (p0) call 0x20000 <myfn_b>
# CHECK-NEXT:     if (!p0) call 0x20054 <__hexagon_thunk_myfn_a_from_.text.thunk>
# CHECK-NEXT:     r2 = add(r0,r1) }

    { r2 = add(r0, r1)
      if (p0) call #myfn_a
      if (!p0) call #myfn_a }
# CHECK-NEXT:   { if (p0) call 0x20054 <__hexagon_thunk_myfn_a_from_.text.thunk>
# CHECK-NEXT:     if (!p0) call 0x20054 <__hexagon_thunk_myfn_a_from_.text.thunk>
# CHECK-NEXT:     r2 = add(r0,r1) }

    { r2 = add(r0, r1)
      r1 = r4
      r4 = r5
      if (r0 == #0) jump:t #myfn_a }
# CHECK-NEXT:   { if (r0==#0) jump:t 0x20054
# CHECK-NEXT:     r2 = add(r0,r1)
# CHECK-NEXT:     r1 = r4; r4 = r5 }

    { r2 = add(r0, r1)
      r4 = r5
      if (r0 <= #0) jump:t #myfn_a
      p1 = cmp.eq(r0, #0); if (p1.new) jump:nt #myfn_a }
# CHECK-NEXT:   { if (r0<=#0) jump:t 0x20054
# CHECK-NEXT:     p1 = cmp.eq(r0,#0x0); if (p1.new) jump:nt 0x20054 <__hexagon_thunk_myfn_a_from_.text.thunk>
# CHECK-NEXT:     r2 = add(r0,r1)
# CHECK-NEXT:     r4 = r5 }

    {r0 = #0; jump #myfn_a}
# CHECK-NEXT:   { r0 = #0x0 ; jump 0x20054 <__hexagon_thunk_myfn_a_from_.text.thunk> }
    {r0 = #0; jump #myfn_b}
# CHECK-NEXT:   { r0 = #0x0 ; jump 0x20000 <myfn_b> }
    jumpr r31
    .size   main, .-main

    .section .text.foo

    .globl myfn_a
    .type  myfn_a, @function
myfn_a:
    {r0 = #0; jump #myfn_b}
    jumpr r31
    .size  myfn_a, .-myfn_a

# CHECK:             00020054 <__hexagon_thunk_myfn_a_from_.text.thunk>:
# CHECK-NONPIC-NEXT: { immext(#0xffff80)
# CHECK-NONPIC-NEXT:   jump 0x1020000 <myfn_a> }
# CHECK-PIC-NEXT:    { immext(#0xffff80)
# CHECK-PIC-NEXT:      r14 = add(pc,##0xffffac) }
# CHECK-PIC-NEXT:    { jumpr r14 }

# CHECK:             01020000 <myfn_a>:
# CHECK-NEXT:        { r0 = #0x0 ; jump 0x1020008 <__hexagon_thunk_myfn_b_from_.text.thunk> }
# CHECK-NEXT:        { jumpr r31 }

# CHECK:             01020008 <__hexagon_thunk_myfn_b_from_.text.thunk>:
# CHECK-NONPIC-NEXT: { immext(#0xfeffffc0)
# CHECK-NONPIC-NEXT:   jump 0x20000 <myfn_b> }
# CHECK-PIC-NEXT:    { immext(#0xfeffffc0)
# CHECK-PIC-NEXT:      r14 = add(pc,##0xfefffff8) }
# CHECK-PIC-NEXT:    { jumpr r14 }
