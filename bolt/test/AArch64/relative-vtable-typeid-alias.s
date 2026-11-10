## Check that relative vtable slots covered by a WPD __typeid_*_unique_member
## alias are still handled as vtable entries. The alias starts at the address
## point and inherits the vtable's size, so it extends past the vtable and is
## the innermost data symbol for the slots.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-linux %s -o %t.o
# RUN: %clang %cflags -shared %t.o -o %t.so -Wl,-q
# RUN: llvm-bolt %t.so -o %t.bolt --lite=0 2>&1 | FileCheck %s
# RUN: llvm-nm %t.bolt > %t.out
# RUN: llvm-objdump -dz -j .rodata %t.bolt >> %t.out
# RUN: FileCheck %s --check-prefix=CHECK-SLOT < %t.out

# CHECK: enabling relocation mode
# CHECK-NOT: unclaimed data relocation

# CHECK-SLOT-DAG: [[#%x,F:]] {{[tT]}} f
# CHECK-SLOT-DAG: [[#%x,VT:]] {{[rR]}} _ZTV1A
# CHECK-SLOT:      <_ZTV1A>:
# CHECK-SLOT-NEXT: .word 0x00000000
# CHECK-SLOT-NEXT: .word 0x00000000
# CHECK-SLOT:      <__typeid__ZTS1A_8_unique_member>:
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
  adrp x0, _ZTV1A
  add x0, x0, :lo12:_ZTV1A
  ret
  .size g, .-g

## A relative vtable: offset-to-top and RTTI (both 0), then two 4-byte virtual
## function slots starting at the address point (_ZTV1A+8). Both slots point to
## f: each holds f minus the address point, so the second slot's relocation is
## f+4.
  .section .rodata,"a",@progbits
  .globl _ZTV1A
  .hidden _ZTV1A
  .type _ZTV1A,@object
  .p2align 2
_ZTV1A:
  .word 0
  .word 0
  .word %pltpcrel(f)
  .word %pltpcrel(f+4)
  .size _ZTV1A, .-_ZTV1A

## No .size, so the alias takes the size of _ZTV1A.
  .globl __typeid__ZTS1A_8_unique_member
  .hidden __typeid__ZTS1A_8_unique_member
  .type __typeid__ZTS1A_8_unique_member,@object
  .set __typeid__ZTS1A_8_unique_member, _ZTV1A+8

## Keep the alias inside .rodata.
  .globl d
  .hidden d
  .type d,@object
d:
  .quad 0
  .size d, 8
