## Verify that BOLT can fully rewrite conditional dealloc_return
## instructions. The disassembler produces pseudo opcodes like
## L4_return_map_to_raw_t and L4_return_map_to_raw_f which lowerToRaw
## must restore to their real forms with implicit D15 and R30 operands.
## Only unconditional dealloc_return is tested in rewrite-allocframe.s.

# RUN: llvm-mc -filetype=obj -triple=hexagon-unknown-linux-musl %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe --emit-relocs -e _start
# RUN: llvm-bolt %t.exe -o %t.bolt 2>&1 | FileCheck --check-prefix=BOLT %s
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

# BOLT-NOT: BOLT-ERROR

# CHECK-LABEL: <func_ret_t>:
# CHECK:       allocframe
# CHECK:       if (p0) dealloc_return

# CHECK-LABEL: <func_ret_f>:
# CHECK:       allocframe
# CHECK:       if (!p0) dealloc_return

  .text
  .globl _start
  .type _start,@function
  .p2align 4
_start:
    call func_ret_t
    call func_ret_f
    jumpr r31
  .size _start, .-_start

  .globl func_ret_t
  .type func_ret_t,@function
  .p2align 4
func_ret_t:
    allocframe(#0)
    p0 = cmp.eq(r0, #0)
    if (p0) dealloc_return
    r0 = #1
    r31:30 = dealloc_return(r30):raw
  .size func_ret_t, .-func_ret_t

  .globl func_ret_f
  .type func_ret_f,@function
  .p2align 4
func_ret_f:
    allocframe(#0)
    p0 = cmp.eq(r0, #0)
    if (!p0) dealloc_return
    r0 = #0
    r31:30 = dealloc_return(r30):raw
  .size func_ret_f, .-func_ret_f
