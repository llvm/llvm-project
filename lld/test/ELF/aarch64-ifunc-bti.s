# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux-gnu %s -o %t.o
# RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux-gnu %p/Inputs/aarch64-addrifunc.s -o %t1.o

# RUN: ld.lld --shared --soname=t1.so %t1.o -o %t1.so
# RUN: ld.lld --pie %t1.so %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn --mattr=+bti --triple=aarch64-linux-gnu %t | FileCheck %s

# RUN: ld.lld -shared -Bsymbolic %t1.so %t.o -o %t.so
# RUN: llvm-objdump -d --no-show-raw-insn --mattr=+bti %t.so | FileCheck %s --check-prefix=SHARED

# When the address of an ifunc is taken using a non-got reference which clang
# can do, LLD exports a canonical PLT entry that may have its address taken so
# we must use bti c.

# CHECK: Disassembly of section .plt:
# CHECK: 00000000000103a0 <.plt>:
# CHECK-NEXT:    103a0:         bti     c
# CHECK-NEXT:                   stp     x16, x30, [sp, #-16]!
# CHECK-NEXT:                   adrp    x16, 0x30000
# CHECK-NEXT:                   ldr     x17, [x16, #1344]
# CHECK-NEXT:                   add     x16, x16, #1344
# CHECK-NEXT:                   br      x17
# CHECK-NEXT:                   nop
# CHECK-NEXT:                   nop
# CHECK: 00000000000103c0 <func1@plt>:
# CHECK-NEXT:    103c0:         adrp    x16, 0x30000
# CHECK-NEXT:                   ldr     x17, [x16, #1352]
# CHECK-NEXT:                   add     x16, x16, #1352
# CHECK-NEXT:                   br      x17
# CHECK-NEXT:                   nop
# CHECK-NEXT:                   nop
# CHECK-EMPTY:
# CHECK: Disassembly of section .iplt:
# CHECK-EMPTY:
## The address of ifunc1@plt does not escape so it does not need `bti c`,
## but having bti is not wrong.
# CHECK-NEXT: 00000000000103e0 <ifunc2>:
# CHECK-NEXT:    103e0:         bti     c
# CHECK-NEXT:                   adrp    x16, 0x30000
# CHECK-NEXT:                   ldr     x17, [x16, #1360]
# CHECK-NEXT:                   add     x16, x16, #1360
# CHECK-NEXT:                   br      x17
# CHECK-NEXT:                   nop
# CHECK-NEXT:    103f8:         bti     c
# CHECK-NEXT:                   adrp    x16, 0x30000
# CHECK-NEXT:                   ldr     x17, [x16, #1368]
# CHECK-NEXT:                   add     x16, x16, #1368
# CHECK-NEXT:                   br      x17
# CHECK-NEXT:                   nop

## The address of ifunc2 (STT_FUNC) escapes, so it must have `bti c`.
# SHARED:      <ifunc2>:
# SHARED-NEXT:    bti     c

# SHARED:         nop
# SHARED-NEXT:    bti     c

.section ".note.gnu.property", "a"
.long 4
.long 0x10
.long 0x5
.asciz "GNU"

.long 0xc0000000 // GNU_PROPERTY_AARCH64_FEATURE_1_AND
.long 4
.long 1          // GNU_PROPERTY_AARCH64_FEATURE_1_BTI
.long 0

.text
.globl ifunc1
.type ifunc1,@gnu_indirect_function
ifunc1:
 ret

.globl ifunc2
.type ifunc2,@gnu_indirect_function
ifunc2:
  ret

.globl func1

.text
.globl _start
.type _start, %function
_start:
  bl func1
  bl ifunc1
  adrp x8, ifunc2
  add x8, x8, :lo12:ifunc2
  ret
