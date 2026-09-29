## Test case for ARM Cortex-A53 erratum 843419 workaround veneers.
## LLD emits veneers when linking with --fix-cortex-a53-843419: the erratum
## sequence is ADRP at 0xff8/0xffc, a qualifying load/store, an optional
## non-branch, then a load/store (unsigned immediate) using the ADRP register;
## LLD replaces that final instruction with B to __CortexA53843419_*.
## BOLT recognizes those veneers (same shape as e843419* stubs) and inlines.
##
## Layout follows lld/test/ELF/aarch64-cortex-a53-843419-address.s test cases.

# RUN: split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-linux-gnu %t/input.s -o %t.o
# RUN: %clang %cflags -fPIC -pie %t.o -o %t.exe -nostdlib \
# RUN:   -fuse-ld=lld -Wl,-q -Wl,-T,%t/linker-script -Wl,--fix-cortex-a53-843419
# RUN: not llvm-bolt %t.exe -o %t.noflag.bolt 2>&1 | FileCheck %s --check-prefix=NOFLAG
# RUN: llvm-bolt %t.exe -o %t.bolt --drop-cortex-a53-843419-veneers
# RUN: llvm-objdump -d %t.bolt | FileCheck %t/input.s

# NOFLAG: BOLT-ERROR: binary contains Cortex-A53 erratum 843419 workaround veneers
# NOFLAG: --drop-cortex-a53-843419-veneers

#--- linker-script
SECTIONS {
  . = SIZEOF_HEADERS;
  .text : { *(.text.main) *(.text.pats) }
  .data : { *(.data.main) }
}

#--- input.s
.section .text.main, "ax", %progbits
.balign 4
.global target_function
.type target_function, %function
target_function:
  ret
.size target_function, .-target_function

## Anti-pattern: manual 3-instruction "veneer" (not LLD's 2-instruction patch).
## BOLT must not inline this.
.global test_pattern_anti
.type test_pattern_anti, %function
test_pattern_anti:
# CHECK-LABEL: <test_pattern_anti>:
  adrp    x8, target_function
  b       e843419_veneer_anti
# CHECK:      b {{.*}} <e843419_veneer_anti>
.Lreturn_anti:
  blr     x8
  ret
.size test_pattern_anti, .-test_pattern_anti

.global e843419_veneer_anti
.type e843419_veneer_anti, %function
e843419_veneer_anti:
# CHECK: <e843419_veneer_anti>:
  ldr     x8, [x8, #0x100]
  add     x8, x8, #0
  b       .Lreturn_anti
.size e843419_veneer_anti, .-e843419_veneer_anti

.section .text.pats, "ax", %progbits

.balign 4096
.space 4096 - 8
.global test_pattern_1
.type test_pattern_1, %function
test_pattern_1:
# CHECK-LABEL: <test_pattern_1>:
  adrp    x7, dat
# CHECK:      adrp
  stp     w9, w8, [sp, #0xc4]
  ldr     x7, [x7, :got_lo12:dat]
# CHECK-NOT:  b {{.*}} <__CortexA53843419_
.Lreturn_1:
  stur    d0, [sp, #0xbc]
  blr     x7
# CHECK:      blr
  ret
.size test_pattern_1, .-test_pattern_1

.balign 4096
.space 4096 - 8
.global test_pattern_2
.type test_pattern_2, %function
test_pattern_2:
# CHECK-LABEL: <test_pattern_2>:
  adrp    x13, dat
# CHECK:      adrp
  ldr     w14, [x10]
  mov     w10, #0x1
  ldr     d0, [x13, :got_lo12:dat]
# CHECK-NOT:  b {{.*}} <__CortexA53843419_
.Lreturn_2:
  mov     x13, #0x220
# CHECK:      mov
  cmp     w14, #0x2ed
  ret
.size test_pattern_2, .-test_pattern_2

.balign 4096
.space 4096 - 8
.global test_pattern_3
.type test_pattern_3, %function
test_pattern_3:
# CHECK-LABEL: <test_pattern_3>:
  adrp    x15, dat
# CHECK:      adrp
  ldr     w0, [sp, #0x20]
  ldr     x15, [x15, :got_lo12:dat]
# CHECK-NOT:  b {{.*}} <__CortexA53843419_
.Lreturn_3:
  add     x15, x15, #0x100
# CHECK:      add
  ret
.size test_pattern_3, .-test_pattern_3

.balign 4096
.space 4096 - 4
.global test_pattern_4
.type test_pattern_4, %function
test_pattern_4:
# CHECK-LABEL: <test_pattern_4>:
  adrp    x8, dat
# CHECK:      adrp
  str     x9, [sp, #0x10]
  ldr     x8, [x8, :got_lo12:dat]
# CHECK-NOT:  b {{.*}} <__CortexA53843419_
.Lreturn_4:
  ldr     x9, [x8, #0x50]
# CHECK:      ldr
  ret
.size test_pattern_4, .-test_pattern_4

.balign 4096
.space 4096 - 4
.global test_pattern_5
.type test_pattern_5, %function
test_pattern_5:
# CHECK-LABEL: <test_pattern_5>:
  adrp    x20, dat
# CHECK:      adrp
  ldr     x2, [sp, #0x20]
  orr     x0, x0, x0
  ldr     x20, [x20, :got_lo12:dat]
# CHECK-NOT:  b {{.*}} <__CortexA53843419_
.Lreturn_5:
  ldrb    w3, [x20, #0x10]
# CHECK:      ldrb
  ret
.size test_pattern_5, .-test_pattern_5

## Pattern 6 (x16): mov x16 + final load; inlining must preserve x16.
.balign 4096
.space 4096 - 4
.global test_pattern_x16
.type test_pattern_x16, %function
test_pattern_x16:
# CHECK-LABEL: <test_pattern_x16>:
  adrp    x1, dat
  ldr     q2, [x0, #3744]
  mov     x16, #0x2600
  ldr     d7, [x1, :got_lo12:dat]
# CHECK-NOT:  b {{.*}} <__CortexA53843419_
.Lreturn_x16:
  adrp    x1, target_function
  ldr     q3, [x3, #0xee0]
# CHECK:      ldr{{.*}}d7
# CHECK:      adrp
  ret
.size test_pattern_x16, .-test_pattern_x16

.section .data.main, "aw", %progbits
.balign 8
.global dat
dat:
  .quad 0
