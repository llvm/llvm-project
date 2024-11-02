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
# CHECK:        0:       10000021        adr     x1, #4
# CHECK: <msg>:
# CHECK:        4:
# #4 is the adr immediate value.

.section .R_AARCH64_ADR_PREL_PG_HI21,"ax",@progbits
  adrp x1,mystr
mystr:
  .asciz "blah"
  .size mystr, 4

# PAGE(S + A) - PAGE(P) = PAGE(210136) - PAGE(0x210132) = 0
#
# CHECK: Disassembly of section .R_AARCH64_ADR_PREL_PG_HI21:
# CHECK-EMPTY:
# CHECK-NEXT: <$x.2>:
# CHECK-NEXT:   210132:       90000001        adrp    x1, 0x210000

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
# CHECK-NEXT: <$x.4>:
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
# CHECK-NEXT: <$x.6>:
# CHECK-NEXT:   210144:       f940a77c        ldr     x28, [x27, #328]

.section .SUB,"ax",@progbits
  nop
sub:
  nop

# CHECK: Disassembly of section .SUB:
# CHECK-EMPTY:
# CHECK-NEXT: <$x.8>:
# CHECK-NEXT:   21014c:       d503201f        nop
# CHECK: <sub>:
# CHECK-NEXT:   210150:       d503201f        nop

.section .R_AARCH64_CALL26,"ax",@progbits
call26:
        bl sub

# S = 0x21014c, A = 0x4, P = 0x210154
# R = S + A - P = -0x4 = 0xfffffffc
# (R & 0x0ffffffc) >> 2 = 0x03ffffff
# 0x94000000 | 0x03ffffff = 0x97ffffff
# CHECK: Disassembly of section .R_AARCH64_CALL26:
# CHECK-EMPTY:
# CHECK-NEXT: <call26>:
# CHECK-NEXT:   210154:       97ffffff        bl     0x210150

.section .R_AARCH64_JUMP26,"ax",@progbits
jump26:
        b sub

# S = 0x21014c, A = 0x4, P = 0x210158
# R = S + A - P = -0x8 = 0xfffffff8
# (R & 0x0ffffffc) >> 2 = 0x03fffffe
# 0x14000000 | 0x03fffffe = 0x17fffffe
# CHECK: Disassembly of section .R_AARCH64_JUMP26:
# CHECK-EMPTY:
# CHECK-NEXT: <jump26>:
# CHECK-NEXT:   210158:       17fffffe        b      0x210150

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
# CHECK-NEXT: f280018c      movk  x12, #12
# CHECK-NEXT: f280018c      movk  x12, #12
# CHECK-NEXT: f2a001ad      movk  x13, #13, lsl #16
# CHECK-NEXT: f2a001ad      movk  x13, #13, lsl #16
# CHECK-NEXT: f2c001ce      movk  x14, #14, lsl #32
# CHECK-NEXT: f2c001ce      movk  x14, #14, lsl #32
# CHECK-NEXT: d2e001ef      mov x15, #4222124650659840
# CHECK-NEXT: f2e001f0      movk  x16, #15, lsl #48

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
# CHECK-NEXT: d2800021      mov x1, #1
# CHECK-NEXT: 92800001      mov x1, #-1
# CHECK-NEXT: d2a00042      mov x2, #131072
## -65537 = 0xfffffffffffeffff
# CHECK-NEXT: 92a00022      mov x2, #-65537
## 12884901888 = 0x300000000
# CHECK-NEXT: d2c00063      mov x3, #12884901888
## -8589934593 = #0xfffffffdffffffff
# CHECK-NEXT: 92c00043      mov x3, #-8589934593

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
# CHECK-NEXT: 2101bc: d2800021     mov	x1, #1
# CHECK-NEXT: 2101c0: 92800001     mov	x1, #-1
# CHECK-NEXT: 2101c4: f2800021     movk	x1, #1
# CHECK-NEXT: 2101c8: f29fffe1     movk	x1, #65535
# CHECK-NEXT: 2101cc: d2a00042     mov	x2, #131072
## -65537 = 0xfffffffffffeffff
# CHECK-NEXT: 2101d0: 92a00022     mov	x2, #-65537
# CHECK-NEXT: 2101d4: f2a00042     movk	x2, #2, lsl #16
# CHECK-NEXT: 2101d8: f2bfffc2     movk	x2, #65534, lsl #16
## 12884901888 = 0x300000000
# CHECK-NEXT: 2101dc: d2c00063     mov	x3, #12884901888
## -8589934593 = #0xfffffffdffffffff
# CHECK-NEXT: 2101e0: 92c00043     mov	x3, #-8589934593
# CHECK-NEXT: 2101e4: f2c00063     movk	x3, #3, lsl #32
# CHECK-NEXT: 2101e8: f2dfffa3     movk	x3, #65533, lsl #32
# CHECK-NEXT: 2101ec: d2c00063     mov	x3, #12884901888
## 1125899906842624 = 0x4000000000000
# CHECK-NEXT: 2101f0: d2e00084     mov	x4, #1125899906842624
# CHECK-NEXT: 2101f4: d2ffff84     mov	x4, #-1125899906842624
# CHECK-NEXT: 2101f8: f2e00084     movk	x4, #4, lsl #48
# CHECK-NEXT: 2101fc: f2ffff84     movk	x4, #65532, lsl #48
