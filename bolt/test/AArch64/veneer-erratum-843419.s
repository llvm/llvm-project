## Test case for ARM Cortex-A53 erratum 843419 workaround veneers
## These veneers have a specific pattern:
## 1. ADRP instruction to load page address
## 2. Optional instructions
## 3. Branch to veneer (e843419@...)
## 4. Veneer contains: LDR/STR using same register, branch back to +1 instruction

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -fPIC -pie %t.o -o %t.exe -nostdlib \
# RUN:    -fuse-ld=lld -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt
# RUN: llvm-objdump -d %t.bolt | FileCheck %s

.text
.balign 4
.global target_function
.type target_function, %function
target_function:
  ret
.size target_function, .-target_function

## Pattern 1: ADRP + multiple instructions + branch to veneer
.global test_pattern_1
.type test_pattern_1, %function
test_pattern_1:
# CHECK-LABEL: <test_pattern_1>:
  adrp    x7, target_function
# CHECK:      adrp
  stp     w9, w8, [sp, #0xc4]
  b       e843419_veneer_1
.Lreturn_1:
  stur    d0, [sp, #0xbc]
  blr     x7
# CHECK-NOT:  b {{.*}} <e843419_veneer_1>
# CHECK:      blr
  ret
.size test_pattern_1, .-test_pattern_1

## Veneer for pattern 1
.global e843419_veneer_1
.type e843419_veneer_1, %function
e843419_veneer_1:
# CHECK-NOT: <e843419_veneer_1>:
  ldr     x7, [x7, #0x1a8]
  b       .Lreturn_1
.size e843419_veneer_1, .-e843419_veneer_1

## Pattern 2: ADRP + load + mov + branch to veneer
.global test_pattern_2
.type test_pattern_2, %function
test_pattern_2:
# CHECK-LABEL: <test_pattern_2>:
  adrp    x13, target_function
# CHECK:      adrp
  ldr     w14, [x10]
  mov     w10, #0x1
  b       e843419_veneer_2
.Lreturn_2:
  mov     x13, #0x220
# CHECK-NOT:  b {{.*}} <e843419_veneer_2>
# CHECK:      mov
  cmp     w14, #0x2ed
  ret
.size test_pattern_2, .-test_pattern_2

## Veneer for pattern 2
.global e843419_veneer_2
.type e843419_veneer_2, %function
e843419_veneer_2:
# CHECK-NOT: <e843419_veneer_2>:
  ldr     d0, [x13, #0x888]
  b       .Lreturn_2
.size e843419_veneer_2, .-e843419_veneer_2

## Pattern 3: Minimal case with just ADRP and branch
.global test_pattern_3
.type test_pattern_3, %function
test_pattern_3:
# CHECK-LABEL: <test_pattern_3>:
  adrp    x15, target_function
# CHECK:      adrp
  b       e843419_veneer_3
.Lreturn_3:
  add     x15, x15, #0x100
# CHECK-NOT:  b {{.*}} <e843419_veneer_3>
# CHECK:      add
  ret
.size test_pattern_3, .-test_pattern_3

## Veneer for pattern 3
.global e843419_veneer_3
.type e843419_veneer_3, %function
e843419_veneer_3:
# CHECK-NOT: <e843419_veneer_3>:
  ldr     x15, [x15, #0x200]
  b       .Lreturn_3
.size e843419_veneer_3, .-e843419_veneer_3

## Pattern 4: Using different registers (x8)
.global test_pattern_4
.type test_pattern_4, %function
test_pattern_4:
# CHECK-LABEL: <test_pattern_4>:
  adrp    x8, target_function
# CHECK:      adrp
  str     x9, [sp, #0x10]
  b       e843419_veneer_4
.Lreturn_4:
  ldr     x9, [x8, #0x50]
# CHECK-NOT:  b {{.*}} <e843419_veneer_4>
# CHECK:      ldr
  ret
.size test_pattern_4, .-test_pattern_4

## Veneer for pattern 4
.global e843419_veneer_4
.type e843419_veneer_4, %function
e843419_veneer_4:
# CHECK-NOT: <e843419_veneer_4>:
  ldr     x8, [x8, #0x400]
  b       .Lreturn_4
.size e843419_veneer_4, .-e843419_veneer_4

## Pattern 5: Complex pattern with conditional code
.global test_pattern_5
.type test_pattern_5, %function
test_pattern_5:
# CHECK-LABEL: <test_pattern_5>:
  adrp    x20, target_function
# CHECK:      adrp
  cbz     x1, .Lskip
  ldr     x2, [sp, #0x20]
.Lskip:
  b       e843419_veneer_5
.Lreturn_5:
  ldrb    w3, [x20, #0x10]
# CHECK-NOT:  b {{.*}} <e843419_veneer_5>
# CHECK:      ldrb
  ret
.size test_pattern_5, .-test_pattern_5

## Veneer for pattern 5
.global e843419_veneer_5
.type e843419_veneer_5, %function
e843419_veneer_5:
# CHECK-NOT: <e843419_veneer_5>:
  ldr     x20, [x20, #0x600]
  b       .Lreturn_5
.size e843419_veneer_5, .-e843419_veneer_5

## Pattern 6 (x16): mov x16 + branch to veneer; inlining preserves x16 for later use.
## Without inlining, LongJmp could turn the veneer into adrp/add/br x16 and clobber it.
.global test_pattern_x16
.type test_pattern_x16, %function
test_pattern_x16:
# CHECK-LABEL: <test_pattern_x16>:
  adrp    x1, target_function
  ldr     q2, [x0, #3744]
  mov     x16, #0x2600
  b       e843419_x16_veneer
.Lreturn_x16:
  adrp    x1, target_function
  ldr     q3, [x3, #0xee0]
# CHECK-NOT:  b {{.*}} <e843419_x16_veneer>
# CHECK:      ldr     q7, [x1, #0xff0]
# CHECK:      adrp
  ret
.size test_pattern_x16, .-test_pattern_x16

.global e843419_x16_veneer
.type e843419_x16_veneer, %function
e843419_x16_veneer:
# CHECK-NOT: <e843419_x16_veneer>:
  ldr     q7, [x1, #0xff0]
  b       .Lreturn_x16
.size e843419_x16_veneer, .-e843419_x16_veneer

## Do NOT inline when target is not the 2-instr e843419 pattern.
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
