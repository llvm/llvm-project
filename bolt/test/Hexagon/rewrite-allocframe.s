## Verify that BOLT can fully rewrite a binary containing allocframe,
## deallocframe, and dealloc_return instructions. These are disassembled
## as map-to-raw pseudo-instructions that must be lowered back to their
## raw forms during encoding. Multiple functions are included so that
## calculateEmittedSize exercises the parallel code path.

# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-linux-musl %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe --emit-relocs -e _start
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck --check-prefix=BOLT %s
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

# BOLT-NOT: BOLT-ERROR

# CHECK-LABEL: <func_a>:
# CHECK:       allocframe
# CHECK:       dealloc_return

# CHECK-LABEL: <func_b>:
# CHECK:       allocframe
# CHECK:       dealloc_return

# CHECK-LABEL: <func_c>:
# CHECK:       allocframe
# CHECK:       deallocframe
# CHECK:       jumpr r31

  .text
  .globl _start
  .type _start,@function
  .p2align 4
_start:
    call func_a
    call func_b
    call func_c
    call func_d
    jumpr r31
  .size _start, .-_start

  .globl func_a
  .type func_a,@function
  .p2align 4
func_a:
    allocframe(#8)
    r0 = #0
    r31:30 = dealloc_return(r30):raw
  .size func_a, .-func_a

  .globl func_b
  .type func_b,@function
  .p2align 4
func_b:
    allocframe(#16)
    r0 = memw(r30 + #0)
    r31:30 = dealloc_return(r30):raw
  .size func_b, .-func_b

  .globl func_c
  .type func_c,@function
  .p2align 4
func_c:
    allocframe(#0)
    r0 = #42
    r31:30 = deallocframe(r30):raw
    jumpr r31
  .size func_c, .-func_c

  .globl func_d
  .type func_d,@function
  .p2align 4
func_d:
    allocframe(#8)
    memw(r30 + #-4) = r0
    r0 = memw(r30 + #-4)
    r31:30 = dealloc_return(r30):raw
  .size func_d, .-func_d
