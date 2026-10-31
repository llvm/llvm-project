## Check that a construction vtable for a class with a non-nested name
## (_ZTC<len><name>...) is handled as a relative vtable: the addend of a vfunc
## slot relocation is not mistaken for an offset into the target function,
## which could make BOLT create an internal reference and mark the function
## non-simple.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-linux %s -o %t.o
# RUN: %clang %cflags -shared %t.o -o %t.so -Wl,-q
# RUN: llvm-bolt %t.so -o %t.bolt --lite=0 2>&1 | FileCheck %s
# RUN: llvm-nm %t.bolt > %t.out
# RUN: llvm-objdump -dz -j .rodata %t.bolt >> %t.out
# RUN: FileCheck %s --check-prefix=CHECK-SLOT < %t.out

# CHECK: enabling relocation mode
# CHECK-NOT: unclaimed data relocation

# CHECK-SLOT-DAG: [[#%x,F:]] {{[tT]}} f
# CHECK-SLOT-DAG: [[#%x,VT:]] {{[rR]}} _ZTC1D0_1B
# CHECK-SLOT:      <_ZTC1D0_1B>:
# CHECK-SLOT-NEXT: .word 0x00000000
# CHECK-SLOT-NEXT: .word 0x00000000
# CHECK-SLOT-NEXT: .word 0x[[#%.8x,F - VT - 8]]
# CHECK-SLOT-NEXT: .word 0x[[#%.8x,F - VT - 8]]

  .text
  .globl f
  .hidden f
  .type f,@function
  .p2align 2
f:
  mov w0, #1
  add w0, w0, #2
  add w0, w0, #3
  ret
  .size f, .-f

  .globl g
  .hidden g
  .type g,@function
  .p2align 2
g:
  adrp x0, _ZTC1D0_1B
  add x0, x0, :lo12:_ZTC1D0_1B
  ret
  .size g, .-g

## Offset-to-top and RTTI, then two slots. A slot's addend is its offset from
## the address point, so the second slot's f+4 lands inside f.
  .section .rodata,"a",@progbits
  .globl _ZTC1D0_1B
  .hidden _ZTC1D0_1B
  .type _ZTC1D0_1B,@object
  .p2align 2
_ZTC1D0_1B:
  .word 0
  .word 0
  .word %pltpcrel(f)
  .word %pltpcrel(f+4)
  .size _ZTC1D0_1B, .-_ZTC1D0_1B
