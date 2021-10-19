# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux-gnu %s -o %t.o
# RUN: ld.lld -shared %t.o -o %tout
# RUN: llvm-objdump -D --no-show-raw-insn %tout | FileCheck %s
# RUN: llvm-readobj -r %tout | FileCheck %s --check-prefix=CHECK-RELOCS

# Test that when we take the address of a preemptible ifunc in a shared object
# we get R_AARCH64_GLOB_DAT to the symbol as it could be defined in another
# link unit and preempt our definition.
.text
.globl myfunc
.type myfunc,@gnu_indirect_function
myfunc:
 ret

.text
.globl main
.type main,@function
main:
 adrp x8, :got:myfunc
 ldr  x8, [x8, :got_lo12:myfunc]
 ret
# CHECK:   0000000000010284 <main>:
## myfunc's got entry = page(0x20330)-page(0x10284) + 0x330 = 65536 + 816
# CHECK-NEXT:    10284: adrp    x8, 0x20000
# CHECK-NEXT:    10288: ldr     x8, [x8, #816]
# CHECK-NEXT:    1028c: ret

# CHECK: Disassembly of section .got:
# CHECK-EMPTY:
# CHECK-NEXT: 0000000000020330 <.got>:

# CHECK-RELOCS: Relocations [
# CHECK-RELOCS-NEXT:   Section {{.*}} .rela.dyn {
# CHECK-RELOCS-NEXT:     0x20330 R_AARCH64_GLOB_DAT myfunc 0x0
# CHECK-RELOCS-NEXT:   }
# CHECK-RELOCS-NEXT: ]
