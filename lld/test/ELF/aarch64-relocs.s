# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-freebsd %s -o %t
# RUN: echo '.globl zero; zero = 0' | llvm-mc -filetype=obj -triple=aarch64-unknown-freebsd -o %t2.o
# RUN: ld.lld %t %t2.o -o %t2
# RUN: llvm-objdump --no-print-imm-hex -d %t2 | FileCheck %s

.section .R_AARCH64_ADR_PREL_LO21,"ax",@progbits
.globl _start
_start:
  adr x1,msg
msg:  .asciz  "Hello, world\n"
msgend:

# CHECK: Disassembly of section .R_AARCH64_ADR_PREL_LO21:
# CHECK-EMPTY:
# CHECK: <_start>:
# CHECK:        0:       10000021        adr     x1, 0x210124
# CHECK: <msg>:
# CHECK:        4:
# #4 is the adr immediate value.

.section .R_AARCH64_ADR_PREL_PG_HI21,"ax",@progbits
  adrp x1,mystr
mystr:
  .asciz "blah"
  .size mystr, 4

# CHECK: Disassembly of section .R_AARCH64_ADR_PREL_PG_HI21:
# CHECK-EMPTY:
# CHECK-NEXT: <.R_AARCH64_ADR_PREL_PG_HI21>:
# CHECK-NEXT:   adrp    x1, 0x210000

.section .R_AARCH64_ADD_ABS_LO12_NC,"ax",@progbits
  add x0, x0, :lo12:.L.str
.L.str:
  .asciz "blah"
  .size mystr, 4

# S = 0x21013b, A = 0x4
# R = (S + A) & 0xFFF = 319
#
# CHECK: Disassembly of section .R_AARCH64_ADD_ABS_LO12_NC:
# CHECK-EMPTY:
# CHECK-NEXT: <.R_AARCH64_ADD_ABS_LO12_NC>:
# CHECK-NEXT:   21013b:       9104fc00        add     x0, x0, #319

.section .R_AARCH64_LDST64_ABS_LO12_NC,"ax",@progbits
  ldr x28, [x27, :lo12:foo]
foo:
  .asciz "foo"
  .size mystr, 3

# S = 0x210144, A = 0x4
# R = ((S + A) & 0xFFF) << 7 = 0x0000a400
# 0x0000a400 | 0xf940177c = 0xf940a77c
# CHECK: Disassembly of section .R_AARCH64_LDST64_ABS_LO12_NC:
# CHECK-EMPTY:
# CHECK-NEXT: <.R_AARCH64_LDST64_ABS_LO12_NC>:
# CHECK-NEXT:   210144:       f940a77c        ldr     x28, [x27, #328]

.section .SUB,"ax",@progbits
  nop
sub:
  nop
.section .R_AARCH64_CALL26,"ax",@progbits
call26:
        bl sub
        b sub

# CHECK: Disassembly of section .R_AARCH64_CALL26:
# CHECK-EMPTY:
# CHECK-NEXT: <call26>:
# CHECK-NEXT:   bl {{.*}} <sub>
# CHECK-NEXT:   b {{.*}} <sub>

.section .R_AARCH64_LDST32_ABS_LO12_NC,"ax",@progbits
ldst32:
  ldr s4, [x5, :lo12:foo32]
foo32:
  .asciz "foo"
  .size mystr, 3

# S = 0x21015c, A = 0x4
# R = ((S + A) & 0xFFC) << 8 = 0x00016000
# 0x00016000 | 0xbd4000a4 = 0xbd4160a4
# CHECK: Disassembly of section .R_AARCH64_LDST32_ABS_LO12_NC:
# CHECK-EMPTY:
# CHECK-NEXT: <ldst32>:
# CHECK-NEXT:   21015c:       bd4160a4        ldr s4, [x5, #352]

.section .R_AARCH64_LDST8_ABS_LO12_NC,"ax",@progbits
ldst8:
  ldrsb x11, [x13, :lo12:foo8]
foo8:
  .asciz "foo"
  .size mystr, 3

# S = 0x210164, A = 0x4
# R = ((S + A) & 0xFFF) << 10 = 0x0005a000
# 0x0005a000 | 0x398001ab = 0x3985a1ab
# CHECK: Disassembly of section .R_AARCH64_LDST8_ABS_LO12_NC:
# CHECK-EMPTY:
# CHECK-NEXT: <ldst8>:
# CHECK-NEXT:   210164:       3985a1ab        ldrsb x11, [x13, #360]

.section .R_AARCH64_LDST128_ABS_LO12_NC,"ax",@progbits
ldst128:
  ldr q20, [x19, #:lo12:foo128]
foo128:
  .asciz "foo"
  .size mystr, 3

# S = 0x21016c, A = 0x4
# R = ((S + A) & 0xFF8) << 6 = 0x00005c00
# 0x00005c00 | 0x3dc00274 = 0x3dc05e74
# CHECK: Disassembly of section .R_AARCH64_LDST128_ABS_LO12_NC:
# CHECK-EMPTY:
# CHECK: <ldst128>:
# CHECK:   21016c:       3dc05e74        ldr     q20, [x19, #368]
#foo128:
#   210170:       66 6f 6f 00     .word

.section .R_AARCH64_LDST16_ABS_LO12_NC,"ax",@progbits
ldst16:
  ldr h17, [x19, :lo12:foo16]
  ldrh w1, [x19, :lo12:foo16]
  ldrh w2, [x19, :lo12:foo16 + 2]
foo16:
  .asciz "foo"
  .size mystr, 4

# S = 0x210174, A = 0x4
# R = ((S + A) & 0x0FFC) << 9 = 0x2f000
# 0x2f000 | 0x7d400271 = 0x7d430271
# CHECK: Disassembly of section .R_AARCH64_LDST16_ABS_LO12_NC:
# CHECK-EMPTY:
# CHECK-NEXT: <ldst16>:
# CHECK-NEXT:   210174:       7d430271        ldr     h17, [x19, #384]
# CHECK-NEXT:   210178:       79430261        ldrh    w1, [x19, #384]
# CHECK-NEXT:   21017c:       79430662        ldrh    w2, [x19, #386]

.section .R_AARCH64_MOVW_UABS,"ax",@progbits
movz1:
   movk x12, #:abs_g0:zero+0xC
   movk x12, #:abs_g0_nc:zero+0xF000E000D000C
   movk x13, #:abs_g1:zero+0xD000C
   movk x13, #:abs_g1_nc:zero+0xF000E000D000C
   movk x14, #:abs_g2:zero+0xE000D000C
   movk x14, #:abs_g2_nc:zero+0xF000E000D000C
   movz x15, #:abs_g3:zero+0xF000E000D000C
   movk x16, #:abs_g3:zero+0xF000E000D000C

## 4222124650659840 == (0xF << 48)
# CHECK: Disassembly of section .R_AARCH64_MOVW_UABS:
# CHECK-EMPTY:
# CHECK-NEXT: <movz1>:
# CHECK-NEXT:   movk  x12, #12
# CHECK-NEXT:   movk  x12, #12
# CHECK-NEXT:   movk  x13, #13, lsl #16
# CHECK-NEXT:   movk  x13, #13, lsl #16
# CHECK-NEXT:   movk  x14, #14, lsl #32
# CHECK-NEXT:   movk  x14, #14, lsl #32
# CHECK-NEXT:   mov x15, #4222124650659840
# CHECK-NEXT:   movk  x16, #15, lsl #48

.section .R_AARCH64_MOVW_SABS,"ax",@progbits
   movz x1, #:abs_g0_s:zero+1
   movz x1, #:abs_g0_s:zero-1
   movz x2, #:abs_g1_s:zero+0x20000
   movz x2, #:abs_g1_s:zero-0x20000
   movz x3, #:abs_g2_s:zero+0x300000000
   movz x3, #:abs_g2_s:zero-0x300000000

# CHECK: Disassembly of section .R_AARCH64_MOVW_SABS:
# CHECK-EMPTY:
# CHECK-NEXT: :
# CHECK-NEXT:   mov x1, #1
# CHECK-NEXT:   mov x1, #-1
# CHECK-NEXT:   mov x2, #131072
## -65537 = 0xfffffffffffeffff
# CHECK-NEXT:   mov x2, #-65537
## 12884901888 = 0x300000000
# CHECK-NEXT:   mov x3, #12884901888
## -8589934593 = #0xfffffffdffffffff
# CHECK-NEXT:   mov x3, #-8589934593

.section .R_AARCH64_MOVW_PREL,"ax",@progbits
   movz x1, #:prel_g0:.+1
   movz x1, #:prel_g0_nc:.-1
   movk x1, #:prel_g0:.+1
   movk x1, #:prel_g0_nc:.-1
   movz x2, #:prel_g1:.+0x20000
   movz x2, #:prel_g1_nc:.-0x20000
   movk x2, #:prel_g1:.+0x20000
   movk x2, #:prel_g1_nc:.-0x20000
   movz x3, #:prel_g2:.+0x300000000
   movz x3, #:prel_g2_nc:.-0x300000000
   movk x3, #:prel_g2:.+0x300000000
   movk x3, #:prel_g2_nc:.-0x300000000
   movz x3, #:prel_g2:.+0x300000000
   movz x4, #:prel_g3:.+0x4000000000000
   movz x4, #:prel_g3:.-0x4000000000000
   movk x4, #:prel_g3:.+0x4000000000000
   movk x4, #:prel_g3:.-0x4000000000000

# CHECK: Disassembly of section .R_AARCH64_MOVW_PREL:
# CHECK-EMPTY:
# CHECK-NEXT: :
# CHECK-NEXT:   mov	x1, #1
# CHECK-NEXT:   mov	x1, #-1
# CHECK-NEXT:   movk	x1, #1
# CHECK-NEXT:   movk	x1, #65535
# CHECK-NEXT:   mov	x2, #131072
## -65537 = 0xfffffffffffeffff
# CHECK-NEXT:   mov	x2, #-65537
# CHECK-NEXT:   movk	x2, #2, lsl #16
# CHECK-NEXT:   movk	x2, #65534, lsl #16
## 12884901888 = 0x300000000
# CHECK-NEXT:   mov	x3, #12884901888
## -8589934593 = #0xfffffffdffffffff
# CHECK-NEXT:   mov	x3, #-8589934593
# CHECK-NEXT:   movk	x3, #3, lsl #32
# CHECK-NEXT:   movk	x3, #65533, lsl #32
# CHECK-NEXT:   mov	x3, #12884901888
## 1125899906842624 = 0x4000000000000
# CHECK-NEXT:   mov	x4, #1125899906842624
# CHECK-NEXT:   mov	x4, #-1125899906842624
# CHECK-NEXT:   movk	x4, #4, lsl #48
# CHECK-NEXT:   movk	x4, #65532, lsl #48
