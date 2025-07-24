# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-linux-musl %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d %t 2>&1 | \
# RUN:     FileCheck --check-prefixes=CHECK-NONPIC,CHECK %s
# RUN: llvm-mc -filetype=obj \
# RUN:         -triple=hexagon-unknown-linux-musl %s -o %t.o
# RUN: ld.lld --pie %t.o -o %t
# RUN: llvm-objdump -d %t 2>&1 | \
# RUN:     FileCheck --check-prefixes=CHECK-PIC,CHECK %s

## Packets with pc-relative relocations are more interesting because
## the offset must be relative to the start of the source, destination
## packets and not necessarily the instruction word containing the jump/call.

# CHECK:  Disassembly of section .text:

# CHECK-NONPIC: 000200b4 <__hexagon_thunk_myfn_a_from_.text.thunk>:
# CHECK-NONPIC: { immext(#0x1000040)
# CHECK-NONPIC:   jump 0x1020110 <myfn_a> }

# CHECK-PIC:    00010150 <__hexagon_thunk_myfn_a_from_.text.thunk>:
# CHECK-PIC-NEXT:    { immext(#0x1000040)
# CHECK-PIC-NEXT:      r14 = add(pc,##0x1000060) }
# CHECK-PIC-NEXT:    { jumpr r14 }

# CHECK-NONPIC: 000200bc <myfn_b>:
# CHECK-NONPIC: { jumpr r31 }
# CHECK-PIC:    0001015c <myfn_b>:
# CHECK-PIC:    { jumpr r31 }
    .globl myfn_b
    .type  myfn_b, @function
myfn_b:
    jumpr r31
    .size  myfn_b, .-myfn_b

# CHECK-PIC:    00010160 <main>:
    .globl main
    .type  main, @function
main:
    { r0 = #0
      call myfn_a }
# CHECK-PIC:      { call 0x10150 <__hexagon_thunk_myfn_a_from_.text.thunk>
# CHECK-NONPIC:   { call 0x200b4 <__hexagon_thunk_myfn_a_from_.text.thunk>
# CHECK-NEXT:       r0 = #0x0 }
    call myfn_a
# CHECK-PIC:    call 0x10150 <__hexagon_thunk_myfn_a_from_.text.thunk>
# CHECK-NONPIC: call 0x200b4 <__hexagon_thunk_myfn_a_from_.text.thunk>
    call myfn_b
# CHECK-PIC-NEXT:    call 0x1015c <myfn_b>
# CHECK-NONPIC-NEXT: call 0x200bc <myfn_b>

    { r2 = add(r0, r1)
      if (p0) call #myfn_b
      if (!p0) call #myfn_a }
# CHECK-PIC-NEXT:     { if (p0) call 0x1015c <myfn_b>
# CHECK-PIC-NEXT:       if (!p0) call 0x10150 <__hexagon_thunk_myfn_a_from_.text.thunk>
# CHECK-NONPIC-NEXT:  { if (p0) call 0x200bc <myfn_b>
# CHECK-NONPIC-NEXT:    if (!p0) call 0x200b4 <__hexagon_thunk_myfn_a_from_.text.thunk>

# CHECK-NEXT:       r2 = add(r0,r1) }

    { r2 = add(r0, r1)
      if (p0) call #myfn_a
      if (!p0) call #myfn_a }
# CHECK-PIC-NEXT:  { if (p0) call 0x10150 <__hexagon_thunk_myfn_a_from_.text.thunk>
# CHECK-PIC-NEXT:    if (!p0) call 0x10150 <__hexagon_thunk_myfn_a_from_.text.thunk>
# CHECK-NONPIC-NEXT:  { if (p0) call 0x200b4 <__hexagon_thunk_myfn_a_from_.text.thunk>
# CHECK-NONPIC-NEXT:    if (!p0) call 0x200b4 <__hexagon_thunk_myfn_a_from_.text.thunk>
# CHECK-NEXT:           r2 = add(r0,r1) }

    { r2 = add(r0, r1)
      r1 = r4
      r4 = r5
      if (r0 == #0) jump:t #myfn_a }
# CHECK-PIC-NEXT:     { if (r0==#0) jump:t 0x10150
# CHECK-NONPIC-NEXT:  { if (r0==#0) jump:t 0x200b4
# CHECK-NEXT:    r2 = add(r0,r1)
# CHECK-NEXT:    r1 = r4; r4 = r5 }

    { r2 = add(r0, r1)
      r4 = r5
      if (r0 <= #0) jump:t #myfn_a
      p1 = cmp.eq(r0, #0); if (p1.new) jump:nt #myfn_a }
# CHECK-NONPIC-NEXT:  { if (r0<=#0) jump:t 0x200b4
# CHECK-NONPIC-NEXT:    p1 = cmp.eq(r0,#0x0); if (p1.new) jump:nt 0x200b4 <__hexagon_thunk_myfn_a_from_.text.thunk>
# CHECK-PIC-NEXT:     { if (r0<=#0) jump:t 0x10150
# CHECK-PIC-NEXT:       p1 = cmp.eq(r0,#0x0); if (p1.new) jump:nt 0x10150 <__hexagon_thunk_myfn_a_from_.text.thunk>
# CHECK-NEXT:           r2 = add(r0,r1)
# CHECK-NEXT:           r4 = r5 }

    {r0 = #0; jump #myfn_a}
# CHECK-PIC-NEXT:    { r0 = #0x0 ; jump 0x10150 <__hexagon_thunk_myfn_a_from_.text.thunk> }
# CHECK-NONPIC-NEXT: { r0 = #0x0 ; jump 0x200b4 <__hexagon_thunk_myfn_a_from_.text.thunk> }
    {r0 = #0; jump #myfn_b}
# CHECK-PIC-NEXT:    { r0 = #0x0 ; jump 0x1015c <myfn_b> }
# CHECK-NONPIC-NEXT: { r0 = #0x0 ; jump 0x200bc <myfn_b> }
    jumpr r31
    .size   main, .-main

    .section .text.foo
    .skip 0x1000000

    .globl myfn_a
    .type  myfn_a, @function
myfn_a:
    {r0 = #0; jump #myfn_b}
    jumpr r31
    .size  myfn_a, .-myfn_a

# CHECK-NONPIC: 01020110 <myfn_a>:
# CHECK-NONPIC-NEXT: { r0 = #0x0 ; jump 0x1020118 <__hexagon_thunk_myfn_b_from_.text.thunk> }
# CHECK-NONPIC-NEXT: { jumpr r31 }

# CHECK-NONPIC: 01020118 <__hexagon_thunk_myfn_b_from_.text.thunk>:
# CHECK-NONPIC-NEXT: { immext(#0xfeffff80)
# CHECK-NONPIC-NEXT:   jump 0x200bc <myfn_b> }

# CHECK-PIC:    010101b8 <__hexagon_thunk_myfn_b_from_.text.thunk>:
# CHECK-PIC-NEXT:    { immext(#0xfeffff80)
# CHECK-PIC-NEXT:      r14 = add(pc,##0xfeffffa4) }
# CHECK-PIC-NEXT:    { jumpr r14 }
