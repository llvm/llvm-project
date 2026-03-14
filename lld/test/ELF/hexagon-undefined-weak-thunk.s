# REQUIRES: hexagon
# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-elf %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

## Undefined weak branch targets are redirected to a linker-synthesized
## guard function containing { jumpr r31 }.  This is safe for all packet
## shapes because the guard is always a valid branch target (packet-
## aligned, single-word packet).  Calls become no-ops (guard returns
## immediately), jumps do an early return.

.weak undefined_weak
.globl _start
.type _start, @function
_start:
  ## Simple call — single-word packet.
  call undefined_weak

  ## Call in a two-word packet with an ALU op.
  { r0 = #0
    call undefined_weak }

  ## Conditional call — single-word packet.
  { if (p0) call #undefined_weak }

  ## Jump in a two-word packet.
  { r0 = #0; jump #undefined_weak }

  ## Two conditional calls plus an ALU op — three-word packet.
  { r2 = add(r0, r1)
    if (p0) call #undefined_weak
    if (!p0) call #undefined_weak }

  ## Conditional jump with ALU op — two-word packet.
  { r2 = add(r0, r1)
    if (r0 == #0) jump:t #undefined_weak }

  jumpr r31

## All branches in _start target the guard function.
# CHECK:      <_start>:
# CHECK-NEXT:   {{[0-9a-f]+}}: { call 0x[[#%x,GUARD:]] <__linker_guard_weak_undef> }
# CHECK-NEXT:   {{[0-9a-f]+}}: { call 0x[[#GUARD]] <__linker_guard_weak_undef>
# CHECK-NEXT:            r0 = #0x0 }
# CHECK-NEXT:   {{[0-9a-f]+}}: { if (p0) call 0x[[#GUARD]] <__linker_guard_weak_undef> }
# CHECK-NEXT:   {{[0-9a-f]+}}: { r0 = #0x0 ; jump 0x[[#GUARD]] <__linker_guard_weak_undef> }
# CHECK-NEXT:   {{[0-9a-f]+}}: { if (p0) call 0x[[#GUARD]] <__linker_guard_weak_undef>
# CHECK-NEXT:            if (!p0) call 0x[[#GUARD]] <__linker_guard_weak_undef>
# CHECK-NEXT:            r2 = add(r0,r1) }
# CHECK-NEXT:   {{[0-9a-f]+}}: { if (r0==#0) jump:t 0x[[#GUARD]]
# CHECK-NEXT:            r2 = add(r0,r1) }
# CHECK-NEXT:          { jumpr r31 }

## The guard section contains a single { jumpr r31 } packet.
# CHECK:      <__linker_guard_weak_undef>:
# CHECK-NEXT:   {{[0-9a-f]+}}: { jumpr r31 }

## No thunks should be created.
# CHECK-NOT: __hexagon_thunk
