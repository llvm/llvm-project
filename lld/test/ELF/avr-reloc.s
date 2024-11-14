; REQUIRES: avr
; RUN: llvm-mc -filetype=obj -triple=avr -mcpu=atmega328p %s -o %t0.o
; RUN: ld.lld %t0.o --defsym=a=0x12345678 --defsym=b=30 --defsym=c=0x15554 -o %t0
; RUN: llvm-objdump -d --print-imm-hex --mcpu=atmega328p %t0 | \
; RUN:     FileCheck --check-prefixes=CHECK,AVR %s
; RUN: llvm-objdump -s --mcpu=atmega328p %t0 | \
; RUN:     FileCheck --check-prefixes=HEX,AVRHEX %s
; RUN: llvm-mc -filetype=obj -triple=avr -mcpu=attiny10 %s --defsym=TINY=1 -o %t1.o
; RUN: ld.lld %t1.o --defsym=a=0x12345678 --defsym=b=30 --defsym=c=0x15554 -o %t1
; RUN: llvm-objdump -d --print-imm-hex --mcpu=attiny10 %t1 | FileCheck %s
; RUN: llvm-objdump -s --mcpu=attiny10 %t1 | \
; RUN:     FileCheck --check-prefixes=HEX,TINYHEX %s

.section .LDI,"ax",@progbits
; CHECK-LABEL: section .LDI:
; CHECK:       ldi     r20, 0x78
; CHECK-NEXT:  ldi     r20, 0x56
; CHECK-NEXT:  ldi     r20, 0x34
; CHECK-NEXT:  ldi     r20, 0x12
; CHECK-NEXT:  ldi     r20, 0x3c
; CHECK-NEXT:  ldi     r20, 0x2b
; CHECK-NEXT:  ldi     r20, 0x1a
; CHECK-NEXT:  ldi     r20, 0xaa
; CHECK-NEXT:  ldi     r20, 0xaa
; CHECK-NEXT:  ldi     r20, 0xff

ldi r20, lo8(a)     ; R_AVR_LO8_LDI
ldi r20, hi8(a)     ; R_AVR_HI8_LDI
ldi r20, hh8(a)     ; R_AVR_HH8_LDI
ldi r20, hhi8(a)    ; R_AVR_MS8_LDI

ldi r20, pm_lo8(a)  ; R_AVR_LO8_LDI_PM
ldi r20, pm_hi8(a)  ; R_AVR_HI8_LDI_PM
ldi r20, pm_hh8(a)  ; R_AVR_HH8_LDI_PM

ldi r20, lo8_gs(c)  ; R_AVR_LO8_LDI_GS
ldi r20, hi8_gs(c)  ; R_AVR_HI8_LDI_GS

ldi r20, b+225

.section .LDI_NEG,"ax",@progbits
; CHECK-LABEL: section .LDI_NEG:
; CHECK:       ldi     r20, 0x88
; CHECK-NEXT:  ldi     r20, 0xa9
; CHECK-NEXT:  ldi     r20, 0xcb
; CHECK-NEXT:  ldi     r20, 0xed
; CHECK-NEXT:  ldi     r20, 0xc4
; CHECK-NEXT:  ldi     r20, 0xd4
; CHECK-NEXT:  ldi     r20, 0xe5
ldi r20, lo8(-(a))     ; R_AVR_LO8_LDI_NEG
ldi r20, hi8(-(a))     ; R_AVR_HI8_LDI_NEG
ldi r20, hh8(-(a))     ; R_AVR_HH8_LDI_NEG
ldi r20, hhi8(-(a))    ; R_AVR_MS8_LDI_NEG

ldi r20, pm_lo8(-(a))  ; R_AVR_LO8_LDI_PM_NEG
ldi r20, pm_hi8(-(a))  ; R_AVR_HI8_LDI_PM_NEG
ldi r20, pm_hh8(-(a))  ; R_AVR_HH8_LDI_PM_NEG

.ifndef TINY
.section .SIX,"ax",@progbits
; AVR-LABEL:    section .SIX:
; AVR:          std   Y+30, r9
; AVR-NEXT:     ldd   r9, Y+30
; AVR-NEXT:     adiw  r24, 0x1e
; AVRHEX-LABEL: section .SIX:
; AVRHEX-NEXT:  9e8e9e8c 4e96
std Y+b, r9   ; R_AVR_6
ldd r9, Y+b   ; R_AVR_6
adiw r24, b   ; R_AVR_6_ADIW
.endif

.section .PORT,"ax",@progbits
; CHECK-LABEL: section .PORT:
; CHECK:       in     r20, 0x1e
; CHECK-NEXT:  sbic   0x1e, 0x1
in    r20, b  ; R_AVR_PORT6
sbic  b, 1    ; R_AVR_PORT5

.section .PCREL,"ax",@progbits
; CHECK-LABEL: section .PCREL
; CHECK:       rjmp .+30
; CHECK-NEXT:  rjmp .-36
; CHECK-NEXT:  breq .+26
; CHECK-NEXT:  breq .-40
; CHECK-NEXT:  rjmp .-4096
; CHECK-NEXT:  rjmp .+4094
; CHECK-NEXT:  rjmp .+4094
; CHECK-NEXT:  rjmp .-4096
; CHECK-NEXT:  breq .-128
; CHECK-NEXT:  breq .+126
; HEX-LABEL:   section .PCREL:
; HEX-NEXT:    0fc0eecf 69f061f3
foo:
rjmp foo + 32  ; R_AVR_13_PCREL
rjmp foo - 32  ; R_AVR_13_PCREL
breq foo + 32  ; R_AVR_7_PCREL
breq foo - 32  ; R_AVR_7_PCREL
rjmp 1f - 4096  $ 1:  ; R_AVR_13_PCREL
rjmp 1f + 4094  $ 1:  ; R_AVR_13_PCREL
rjmp 1f - 4098  $ 1:  ; R_AVR_13_PCREL (overflow)
rjmp 1f + 4096  $ 1:  ; R_AVR_13_PCREL (overflow)
breq 1f - 128   $ 1:  ; R_AVR_7_PCREL
breq 1f + 126   $ 1:  ; R_AVR_7_PCREL

.section .LDSSTS,"ax",@progbits
; CHECK-LABEL: section .LDSSTS:
; CHECK:       lds r20, 0x1e
; CHECK-NEXT:  sts 0x1e, r21
; HEX-LABEL:   section .LDSSTS:
; AVRHEX:      {{.*}} 40911e00 50931e00
; TINYHEX:     {{.*}} 4ea15ea9
lds r20, b
sts b, r21

.section .DATA,"ax",@progbits
; HEX-LABEL: section .DATA:
; HEX-NEXT:  {{.*}} 1e1e000f 00785634 12785634
.byte b        ; R_AVR_8
.short b       ; R_AVR_16
.short gs(b)   ; R_AVR_16_PM
.long a        ; R_AVR_32
.byte lo8(a)   ; R_AVR_8_LO8
.byte hi8(a)   ; R_AVR_8_HI8
.byte hlo8(a)  ; R_AVR_8_HLO8
