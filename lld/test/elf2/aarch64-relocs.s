# RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-freebsd %s -o %t
# RUN: ld.lld2 %t -o %t2
# RUN: llvm-objdump -d %t2 | FileCheck %s
# REQUIRES: aarch64

.section .R_AARCH64_ADR_PREL_LO21,"ax",@progbits
.globl _start
_start:
  adr x1,msg
msg:  .asciz  "Hello, world\n"
msgend:

# CHECK: Disassembly of section .R_AARCH64_ADR_PREL_LO21:
# CHECK: _start:
# CHECK:        0:       21 00 00 10     adr     x1, #4
# CHECK: msg:
# CHECK:        4:
# #4 is the adr immediate value.

.section .R_AARCH64_ADR_PREL_PG_H121,"ax",@progbits
  adrp x1,mystr
mystr:
  .asciz "blah"
  .size mystr, 4

# S = 0x11012, A = 0x4, P = 0x11012
# PAGE(S + A) = 0x11000
# PAGE(P) = 0x11000
#
# CHECK: Disassembly of section .R_AARCH64_ADR_PREL_PG_H121:
# CHECK-NEXT: $x.2:
# CHECK-NEXT:   11012:       01 00 00 90     adrp    x1, #0

.section .R_AARCH64_ADD_ABS_LO12_NC,"ax",@progbits
  add x0, x0, :lo12:.L.str
.L.str:
  .asciz "blah"
  .size mystr, 4

# S = 0x1101b, A = 0x4
# R = (S + A) & 0xFFF = 0x1f
# R << 10 = 0x7c00
#
# CHECK: Disassembly of section .R_AARCH64_ADD_ABS_LO12_NC:
# CHECK-NEXT: $x.4:
# CHECK-NEXT:   1101b:       00 7c 00 91     add     x0, x0, #31

.section .R_AARCH64_LDST64_ABS_LO12_NC,"ax",@progbits
  ldr x28, [x27, :lo12:foo]
foo:
  .asciz "foo"
  .size mystr, 3

# S = 0x11024, A = 0x4
# R = ((S + A) & 0xFFF) << 7 = 0x00001400
# 0x00001400 | 0xf940177c = 0xf940177c
# CHECK: Disassembly of section .R_AARCH64_LDST64_ABS_LO12_NC:
# CHECK-NEXT: $x.6:
# CHECK-NEXT:   11024:       7c 17 40 f9     ldr     x28, [x27, #40]

.section .SUB,"ax",@progbits
  nop
sub:
  nop

# CHECK: Disassembly of section .SUB:
# CHECK-NEXT: $x.8:
# CHECK-NEXT:   1102c:       1f 20 03 d5     nop
# CHECK: sub:
# CHECK-NEXT:   11030:       1f 20 03 d5     nop

.section .R_AARCH64_CALL26,"ax",@progbits
call26:
        bl sub

# S = 0x1102c, A = 0x4, P = 0x11034
# R = S + A - P = -0x4 = 0xfffffffc
# (R & 0x0ffffffc) >> 2 = 0x03ffffff
# 0x94000000 | 0x03ffffff = 0x97ffffff
# CHECK: Disassembly of section .R_AARCH64_CALL26:
# CHECK-NEXT: call26:
# CHECK-NEXT:   11034:       ff ff ff 97     bl     #-4

.section .R_AARCH64_JUMP26,"ax",@progbits
jump26:
        b sub

# S = 0x1102c, A = 0x4, P = 0x11038
# R = S + A - P = -0x8 = 0xfffffff8
# (R & 0x0ffffffc) >> 2 = 0x03fffffe
# 0x14000000 | 0x03fffffe = 0x17fffffe
# CHECK: Disassembly of section .R_AARCH64_JUMP26:
# CHECK-NEXT: jump26:
# CHECK-NEXT:   11038:       fe ff ff 17     b      #-8
