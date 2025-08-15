## This test checks that disassemble stage works properly
## JT with indirect branch
## 1) nop + adr pair instructions
## 2) sub + ldr pair instructions
## 3) adrp + ldr pair instructions
## 4) pic jt with relive offsets packed to 1-byte entry size
## 5) fixed indirect branch
## 6) normal jt

# REQUIRES: system-linux

# RUN: rm -rf %t && split-file %s %t

## Prepare binary (1)
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %t/jt_nop_adr.s \
# RUN:    -o %t/jt_nop_adr.o
# RUN: %clang %cflags --target=aarch64-unknown-linux %t/jt_nop_adr.o \
# RUN:  -Wl,-q -Wl,-z,now, -Wl,-T,%t/within-adr-range.t -o %t/jt_nop_adr.exe
# RUN: llvm-objdump --no-show-raw-insn -d %t/jt_nop_adr.exe | FileCheck \
# RUN:    --check-prefix=JT-RELAXED %s

# JT-RELAXED: <_start>:
# JT-RELAXED-NEXT:  nop
# JT-RELAXED-NEXT:  adr {{.*}}x3

# RUN: llvm-bolt %t/jt_nop_adr.exe -o %t/jt_nop_adr.bolt -v 3 2>&1 | FileCheck \
# RUN:    --check-prefix=JT-BOLT-RELAXED %s

# JT-BOLT-RELAXED: failed to match indirect branch

## This linker script ensures that .rodata and .text are sufficiently (<1M)
## close to each other so that the adrp + ldr pair can be relaxed to nop + adr.
#--- within-adr-range.t
SECTIONS {
 .rodata 0x1000: { *(.rodata) }
 .text   0x2000: { *(.text) }
 .rela.rodata :    { *(.rela.rodata) }
}

## Prepare binary (2)
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %t/jt_sub_ldr.s \
# RUN:   -o %t/jt_sub_ldr.o
# RUN: %clang %cflags --target=aarch64-unknown-linux %t/jt_sub_ldr.o \
# RUN:  -Wl,-q -Wl,-z,now -o %t/jt_sub_ldr.exe
# RUN: llvm-objdump --no-show-raw-insn -d %t/jt_sub_ldr.exe | FileCheck \
# RUN:    --check-prefix=JT-SUB-LDR %s

# JT-SUB-LDR: <_start>:
# JT-SUB-LDR-NEXT:  sub
# JT-SUB-LDR-NEXT:  ldr

# RUN: llvm-bolt %t/jt_sub_ldr.exe -o %t/jt_sub_ldr.bolt -v 3 2>&1 | FileCheck \
# RUN:    --check-prefix=JT-BOLT-SUBLDR %s
# JT-BOLT-SUBLDR: failed to match indirect branch

## Prepare binary (3)
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %t/jt_adrp_ldr.s \
# RUN:    -o %t/jt_adrp_ldr.o
# RUN: %clang %cflags --target=aarch64-unknown-linux %t/jt_adrp_ldr.o \
# RUN:  -Wl,-q -Wl,-z,now  -Wl,--no-relax -o %t/jt_adrp_ldr.exe
# RUN: llvm-objdump --no-show-raw-insn -d %t/jt_adrp_ldr.exe | FileCheck \
# RUN:   --check-prefix=JT-ADRP-LDR %s

# JT-ADRP-LDR: <_start>:
# JT-ADRP-LDR-NEXT:  adrp
# JT-ADRP-LDR-NEXT:  ldr

# RUN: llvm-bolt %t/jt_adrp_ldr.exe -o %t/jt_adrp_ldr.bolt -v 3 2>&1 | FileCheck \
# RUN:   --check-prefix=JT-BOLT-ADRP-LDR %s
# JT-BOLT-ADRP-LDR: failed to match indirect branch

## Prepare binary (4)
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:    --position-independent %t/jt_pic_with_relative_offset.s \
# RUN:    -o %t/jt_pic_with_relative_offset.o
# RUN: %clang %cflags -fPIC -O0  %t/jt_pic_with_relative_offset.o \
# RUN:    -o %t/jt_pic_with_relative_offset.exe -Wl,-q -Wl,--no-relax
# RUN: llvm-bolt %t/jt_pic_with_relative_offset.exe \
# RUN:    -o %t/jt_pic_with_relative_offset.bolt -v 3 2>&1 | FileCheck \
# RUN:   --check-prefix=JT-BOLT-JT-PIC-OFFSETS %s

# JT-BOLT-JT-PIC-OFFSETS: failed to match indirect branch

## Prepare binary (5)
# RUN: %clang %cflags %t/jt_fixed_branch.s -Wl,-q -Wl,--no-relax \
# RUN:     -o %t/jt_fixed_branch.exe

# RUN: llvm-bolt %t/jt_fixed_branch.exe \
# RUN:    -o %t/jt_fixed_branch.bolt -v 3 2>&1 | FileCheck \
# RUN:   --check-prefix=JT-BOLT-FIXED-BR %s

# JT-BOLT-FIXED-BR: failed to match indirect branch

## Prepare binary (6)
# RUN: %clang %cflags -no-pie %t/jt_type_normal.c \
# RUN:   -Wl,-q -Wl,-z,now -Wl,--no-relax \
# RUN:   -o %t/jt_type_normal.exe
# RUN: llvm-objdump --no-show-raw-insn -d %t/jt_type_normal.exe | FileCheck \
# RUN:   --check-prefix=JT-OBJDUMP-NORMAL %s

# JT-OBJDUMP-NORMAL: <handleOptionJumpTable>:
# JT-OBJDUMP-NORMAL:  adrp
# JT-OBJDUMP-NORMAL-NEXT:  add
# JT-OBJDUMP-NORMAL-NEXT:  ldr
# JT-OBJDUMP-NORMAL-NEXT:  blr

# RUN: llvm-bolt %t/jt_type_normal.exe --dyno-stats \
# RUN:    -o %t/jt_type_normal.bolt -v 3 2>&1 | FileCheck \
# RUN:   --check-prefix=JT-BOLT-NORMAL %s

# JT-BOLT-NORMAL: 0{{.*}}: indirect calls

#--- jt_nop_adr.s
  .globl _start
  .type  _start, %function
_start:
  adrp    x3, :got:jump_table
  ldr     x3, [x3, #:got_lo12:jump_table]
  ldrh    w3, [x3, x1, lsl #1]
  adr     x1, test2_0
  add     x3, x1, w3, sxth #2
  br      x3
test2_0:
  ret
test2_1:
  ret

  .section .rodata,"a",@progbits
jump_table:
  .hword  (test2_0-test2_0)>>2
  .hword  (test2_1-test2_0)>>2


#--- jt_sub_ldr.s
  .globl _start
  .type  _start, %function
_start:
  sub     x1, x29, #0x4, lsl #12
  ldr     x1, [x1, #14352]
  ldrh    w1, [x1, w3, uxtw #1]
  adr     x3, test2_0
  add     x1, x3, w1, sxth #2
  br      x1
test2_0:
  ret
test2_1:
  ret

  .section .rodata,"a",@progbits
jump_table:
  .hword  (test2_0-test2_0)>>2
  .hword  (test2_1-test2_0)>>2


#--- jt_adrp_ldr.s
  .globl _start
  .type  _start, %function
_start:
  adrp    x3, :got:jump_table
  ldr     x3, [x3, #:got_lo12:jump_table]
  ldrh    w3, [x3, x1, lsl #1]
  adr     x1, test2_0
  add     x3, x1, w3, sxth #2
  br      x3
test2_0:
  ret
test2_1:
  ret

  .section .rodata,"a",@progbits
jump_table:
  .hword  (test2_0-test2_0)>>2
  .hword  (test2_1-test2_0)>>2


#--- jt_pic_with_relative_offset.s
.text
.global _start
_start:
  mov     x4, 3  // index in jmp table where offset related to adr instr
  adrp    x0, funcTableSym
  add     x0, x0, #:lo12:funcTableSym
  ldrb    w0, [x0, w4, uxtw #0]
  adr     x2, .LBB1
  add     x0, x2, w0, sxth #2
  br      x0

.LBB1:
  bl      funcA
  b       .test_exit

.LBB2:
  bl      funcB
  b       .test_exit

.LBB3:
  bl      funcC
  b       .test_exit

.LBB4:
  bl      funcD
  b       .test_exit

.test_exit:
  mov x8, #93
  mov x0, #0
  svc #0

.global funcA
funcA:
  ret

.global funcB
funcB:
  ret

.global funcC
funcC:
  ret

.global funcD
funcD:
  ret

.section .rodata,"a",@progbits
.align 2
funcTableSym:
  .byte 0x00,0x02,0x04,0x06  // 1 - .LBB1, 3 - .LBB2

#--- jt_fixed_branch.s

.text
.global _start
_start:
  mov x0, x13
  mov x1, x4
  mov x0, x2
  movk x1, #0x0, lsl #48
  movk x1, #0x0, lsl #32
  movk x1, #0x0, lsl #16
  movk x1, #0x12
  stp x0, x1, [sp, #-16]!
  adrp x0, foo
  add  x0, x0, #:lo12:foo
  br   x0
  mov x8, #93
  mov x0, #0
  svc #0

.global foo
.type foo,%function
foo:
  mov x8, #9
  ret
.size foo,.-foo

#--- jt_type_normal.c

void __attribute__ ((noinline)) option0() {
}

void __attribute__ ((noinline)) option1() {
}

void __attribute__ ((noinline)) option2() {
}

void __attribute__ ((noinline)) option3() {
}

void __attribute__ ((noinline)) option4() {
}

void __attribute__ ((noinline)) option5() {
}

void (*jumpTable[6])() = { option0, option1, option2, option3, option4, option5 };

void __attribute__ ((noinline)) handleOptionJumpTable(int option) {
    jumpTable[option]();
}

int main(int argc, char *argv[]) {
    handleOptionJumpTable(argc);
    return 0;
}
